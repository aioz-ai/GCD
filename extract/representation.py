import pathlib
from argparse import ArgumentParser

# imports and set up Jukebox's multi-GPU parallelization
import jukebox
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from accelerate import init_empty_weights
from tqdm import tqdm

import librosa as lr
import numpy as np
import torch
import torch as t
import os
import gc

# --------------------


#@title Jukebox extraction code

###########################
# Jukebox extraction code #
###########################

# Note: this code was written by reverse-engineering the model, which entailed
# combing through https://github.com/openai/jukebox all the way down the stack
# trace together with the readily-executable Colab example https://colab.research.google.com/github/openai/jukebox/blob/master/jukebox/Interacting_with_Jukebox.ipynb
# and modifying values as necessary to get what we needed.

# --- MODEL PARAMS ---
DEPTH = 66
JUKEBOX_SAMPLE_RATE = 44100
T = 8192

# 1048576 found in paper, last page
SAMPLE_LENGTH = 1048576  
DEFAULT_DURATION = SAMPLE_LENGTH / JUKEBOX_SAMPLE_RATE # duration = SAMPLE_LENGTH/SR ~ 23.77s, which is the param in jukebox

VQVAE_RATE = T / DEFAULT_DURATION

def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()

def load_audio_from_file(fpath, offset=0.0, duration=None):
    if duration is not None:
        audio, _ = lr.load(fpath,
                           sr=JUKEBOX_SAMPLE_RATE,
                           offset=offset,
                           duration=duration)
    else:
        audio, _ = lr.load(fpath,
                           sr=JUKEBOX_SAMPLE_RATE,
                           offset=offset)

    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()


def get_z(audio):
    # don't compute unnecessary discrete encodings
    audio = audio[: JUKEBOX_SAMPLE_RATE * 25]

    zs = vqvae.encode(torch.cuda.FloatTensor(audio[np.newaxis, :, np.newaxis]))

    z = zs[-1].flatten()[np.newaxis, :]

    return z


def get_cond(hps, top_prior):
    # model only accepts sample length conditioning of
    # >60 seconds
    sample_length_in_seconds = 62

    hps.sample_length = (
        int(sample_length_in_seconds * hps.sr) // top_prior.raw_to_tokens
    ) * top_prior.raw_to_tokens

    # NOTE: the 'lyrics' parameter is required, which is why it is included,
    # but it doesn't actually change anything about the `x_cond`, `y_cond`,
    # nor the `prime` variables. The `prime` variable is supposed to represent
    # the lyrics, but the LM prior we're using does not condition on lyrics,
    # so it's just an empty tensor.
    metas = [
        dict(
            artist="unknown",
            genre="unknown",
            total_length=hps.sample_length,
            offset=0,
            lyrics="""lyrics go here!!!""",
        ),
    ] * hps.n_samples

    labels = [None, None, top_prior.labeller.get_batch_labels(metas, "cuda")]

    x_cond, y_cond, prime = top_prior.get_cond(None, top_prior.get_y(labels[-1], 0))

    x_cond = x_cond[0, :T][np.newaxis, ...]
    y_cond = y_cond[0][np.newaxis, ...]

    return x_cond, y_cond

def downsample(representation,
               target_rate=30,
               method=None):
    if method is None:
        method = 'librosa_fft'

    if method == 'librosa_kaiser':
        resampled_reps = lr.resample(np.asfortranarray(representation.T),
                                     T / DEFAULT_DURATION,
                                     target_rate).T
    elif method in ['librosa_fft', 'librosa_scipy']:
        resampled_reps = lr.resample(np.asfortranarray(representation.T),
                                     T / DEFAULT_DURATION,
                                     target_rate,
                                     res_type='fft').T
    elif method == 'mean':
        raise NotImplementedError

    return resampled_reps

def get_final_activations(z, x_cond, y_cond, top_prior):

    x = z[:, :T]

    input_length = x.shape[1]

    if x.shape[1] < T:
        # arbitrary choices
        min_token = 0
        max_token = 100

        x = torch.cat((x,
                       torch.randint(min_token, max_token, size=(1, T - input_length,), device='cuda')),
                      dim=-1)

    # encoder_kv and fp16 are set to the defaults, but explicitly so
    out = top_prior.prior.forward(
        x, x_cond=x_cond, y_cond=y_cond, encoder_kv=None, fp16=False
    )

    # chop off, in case input was already chopped
    out = out[:,:input_length]

    return out

def roll(x, n):
    return t.cat((x[:, -n:], x[:, :-n]), dim=1)

def get_activations_custom(x,
                           x_cond,
                           y_cond,
                           layers_to_extract=None,
                           fp16=False,
                           fp16_out=False):

    # this function is adapted from:
    # https://github.com/openai/jukebox/blob/08efbbc1d4ed1a3cef96e08a931944c8b4d63bb3/jukebox/prior/autoregressive.py#L116

    # custom jukemir stuff
    if layers_to_extract is None:
        layers_to_extract = [36]

    x = x[:,:T]  # limit to max context window of Jukebox

    input_seq_length = x.shape[1]

    # chop x_cond if input is short
    x_cond = x_cond[:, :input_seq_length]

    # Preprocess.
    with t.no_grad():
        x = top_prior.prior.preprocess(x)

    N, D = x.shape
    assert isinstance(x, t.cuda.LongTensor)
    assert (0 <= x).all() and (x < top_prior.prior.bins).all()

    if top_prior.prior.y_cond:
        assert y_cond is not None
        assert y_cond.shape == (N, 1, top_prior.prior.width)
    else:
        assert y_cond is None

    if top_prior.prior.x_cond:
        assert x_cond is not None
        assert x_cond.shape == (N, D, top_prior.prior.width) or x_cond.shape == (N, 1, top_prior.prior.width), f"{x_cond.shape} != {(N, D, top_prior.prior.width)} nor {(N, 1, top_prior.prior.width)}. Did you pass the correct --sample_length?"
    else:
        assert x_cond is None
        x_cond = t.zeros((N, 1, top_prior.prior.width), device=x.device, dtype=t.float)

    x_t = x # Target
    # self.x_emb is just a straightforward embedding, no trickery here
    x = top_prior.prior.x_emb(x) # X emb
    # this is to be able to fit in a start token/conditioning info: just shift to the right by 1
    x = roll(x, 1) # Shift by 1, and fill in start token
    # self.y_cond == True always, so we just use y_cond here
    if top_prior.prior.y_cond:
        x[:,0] = y_cond.view(N, top_prior.prior.width)
    else:
        x[:,0] = top_prior.prior.start_token

    # for some reason, p=0.0, so the dropout stuff does absolutely nothing
    x = top_prior.prior.x_emb_dropout(x) + top_prior.prior.pos_emb_dropout(top_prior.prior.pos_emb())[:input_seq_length] + x_cond # Pos emb and dropout

    layers = top_prior.prior.transformer._attn_mods

    reps = {}

    if fp16:
        x = x.half()

    for i, l in enumerate(layers):
        # to be able to take in shorter clips, we set sample to True,
        # but as a consequence the forward function becomes stateful
        # and its state changes when we apply a layer (attention layer
        # stores k/v's to cache), so we need to clear its cache religiously
        l.attn.del_cache()

        x = l(x, encoder_kv=None, sample=True)

        l.attn.del_cache()

        if i + 1 in layers_to_extract:
            reps[i + 1] = np.array(x.squeeze().cpu())

            # break if this is the last one we care about
            if layers_to_extract.index(i + 1) == len(layers_to_extract) - 1:
                break

    return reps


# important, gradient info takes up too much space,
# causes CUDA OOM
@torch.no_grad()
def get_acts_from_audio(audio=None,
                        fpath=None,
                        meanpool=False,
                        # pick which layer(s) to extract from
                        layers=None,
                        # pick which part of the clip to load in
                        offset=0.0,
                        duration=None,
                        # downsampling frame-wise reps
                        downsample_target_rate=None,
                        downsample_method=None,
                        # for speed-saving
                        fp16=False,
                        # for space-saving
                        fp16_out=False,
                        # for GPU VRAM. potentially slows it
                        # down but we clean up garbage VRAM.
                        # disable if your GPU has a lot of memory
                        # or if you're extracting from earlier
                        # layers.
                        force_empty_cache=True):

    # main function that runs extraction end-to-end.

    if layers is None:
        layers = [36]  # by default

    if audio is None:
        assert fpath is not None
        audio = load_audio_from_file(fpath, offset=offset, duration=duration)
    elif fpath is None:
        assert audio is not None

    if force_empty_cache: empty_cache()

    # run vq-vae on the audio to get discretized audio
    z = get_z(audio)

    if force_empty_cache: empty_cache()

    # get conditioning info
    x_cond, y_cond = get_cond(hps, top_prior)

    if force_empty_cache: empty_cache()

    # get the activations from the LM
    acts = get_activations_custom(z,
                                  x_cond,
                                  y_cond,
                                  layers_to_extract=layers,
                                  fp16=fp16,
                                  fp16_out=fp16_out)

    if force_empty_cache: empty_cache()

    # postprocessing
    if downsample_target_rate is not None:
        for num in acts.keys():
            acts[num] = downsample(acts[num],
                                   target_rate=downsample_target_rate,
                                   method=downsample_method)

    if meanpool:
        acts = {num: act.mean(axis=0) for num, act in acts.items()}

    if not fp16_out:
        acts = {num: act.astype(np.float32) for num, act in acts.items()}

    return acts



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="input directory containing .wav files", default="audio_data/custom_music/")
    parser.add_argument("--output_dir", type=str, help="output directory for .npy features to be extracted", default="outputs/custom_music/")
    args = parser.parse_args()


    # --- SETTINGS ---

    DEVICE = 'cuda'
    VQVAE_MODELPATH = "pretrained/5b/vqvae.pth.tar"
    PRIOR_MODELPATH = "pretrained/5b/prior_level_2.pth.tar"


    
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir


    
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    FPS = 30 
    #  Since the output shape is 8192 * 4800, the params bust can divide 8192.
    USING_CACHED_FILE = True
    model_type = "5b"  # might not fit to other settings, e.g., "1b_lyrics" or "5b_lyrics"

    # --- SETTINGS ---
    device = DEVICE
    # Set up VQVAE

    hps = Hyperparams()
    hps.sr = 44100
    hps.n_samples = 8
    hps.name = "samples"
    chunk_size = 32
    max_batch_size = 16
    hps.levels = 3
    hps.hop_fraction = [0.5, 0.5, 0.125]
    vqvae, *priors = MODELS[model_type] #model dict containing name of  the sub-models (vqvae, prior, upsampler,...)


    hps_1 = setup_hparams(vqvae, dict(sample_length=SAMPLE_LENGTH))
    hps_1.restore_vqvae = VQVAE_MODELPATH
    # with init_empty_weights():
    #     vqvae = make_vqvae(hps_1, 'meta') #device)
    vqvae = make_vqvae(hps_1, device)


    # Set up language model (top-level prior decoder)
    hps_2 = setup_hparams(priors[-1], dict())
    hps_2["prior_depth"] = DEPTH # we only need to load the first 36 layers of the transformer
    hps_2.restore_prior = PRIOR_MODELPATH
    # with init_empty_weights():
    #     top_prior = make_prior(hps_2, vqvae, 'meta')#device)
    top_prior = make_prior(hps_2, vqvae, device)


    # ============================ RUN FEATURE EXRACTION =====================================
    input_dir = pathlib.Path(INPUT_DIR)
    output_dir = pathlib.Path(OUTPUT_DIR)
    input_paths = sorted(list(input_dir.iterdir()))
    # filter wav file
    input_paths = list(filter(lambda x: x.name.endswith('.wav') or x.name.endswith('.mp3'), input_paths))

    LOAD_WINDOW_DURATION = 20 # divide the audio signal into non-overlapping window (20s)
 
    for input_path in tqdm(input_paths):
        # Check if output already exists
        output_path = pathlib.Path(output_dir, f"{input_path.stem}.npy")

        if os.path.exists(str(output_path)) and USING_CACHED_FILE:  # load cached data, and skip calculating
            print("Skip exist:", output_path)
            # np.load(output_path)
            continue

        print("Extracting: ", str(input_path))

        offset = 0
        whole_seq_features = []
        while True: # divide the audio signal into non-overlapping window (with 20s) and loop until the end, increasing the offset by seconds
            # Decode, resample, convert to mono, and normalize audio
            audio = load_audio_from_file(input_path, offset = offset, duration = LOAD_WINDOW_DURATION)
            actual_duration = len(audio)/JUKEBOX_SAMPLE_RATE
            if actual_duration * FPS < 1: #if the number of expected frames < 1 then we need no processing
                break
            print("offset:", offset, ", audio: ", audio.shape, ",duration: ", len(audio)/JUKEBOX_SAMPLE_RATE)
            with torch.no_grad():
                acts = get_acts_from_audio(audio=audio,
                                                    layers=[DEPTH],
                                                    meanpool=False,
                                                    downsample_target_rate=FPS)
                representation = acts[DEPTH] #get activations at layer 36
            whole_seq_features.append(representation)
            print("current feature shape:", representation.shape)
            if len(audio) < LOAD_WINDOW_DURATION * JUKEBOX_SAMPLE_RATE: #if we are at the final window (actual duration < expected duration)
                break
            offset += LOAD_WINDOW_DURATION
        whole_seq_features = np.concatenate(whole_seq_features, axis=0 ) #shape (n_frames, 4800) 
        # The audio frames are aligned with the video frames at the first frame (0)
        print("Final Audio feature shape:", whole_seq_features.shape)

        np.save(output_path, whole_seq_features)
