
import os
from librosa import beat
import torch
import numpy as np
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal

import argparse

import glob
import tqdm
from smplx import SMPL




def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
    return keypoints3d


def motion_peak_onehot(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - peak_onhot: motion beats.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=5) # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # # Second-derivative of the velocity shows the energy of the beats
    # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
    # # optimize peaks
    # peak_onehot[peak_energy<0.001] = 0
    return peak_onehot


def alignment_score(music_beats, motion_beats, sigma=2):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all)


def calculate_beat_score_gt(seq_name, motion_dir, audio_features_dir, smpl):
    # get real data motion beats
    data = pkl.load(open(os.path.join(motion_dir, f'{seq_name}.pkl'),"rb"))
    smpl_poses = data['smpl_poses']
    smpl_trans = data['root_trans']

    N, T = smpl_poses.shape[:2]
    J = smpl.NUM_JOINTS  # 23

    global_orient = smpl_poses[:,:,:3].reshape(N, -1, 3)
    body_pose = smpl_poses[:,:,3:].reshape(N, -1, J*3)
    smpl_trans = smpl_trans.reshape(N, -1,3)

    # get real data music beats
    audio_feature = np.load(os.path.join(audio_features_dir, f"{seq_name}.npy"))
    audio_beats = audio_feature[:T, -1]  # last dim is the music beats

    beat_scores = []

    for n in range(N):
        # keypoints3d = all_joints3d[n]
        with torch.no_grad():
            keypoints3d = smpl.forward(
                global_orient=torch.from_numpy(global_orient[n]).float().to(args.device),
                body_pose=torch.from_numpy(body_pose[n]).float().to(args.device),
                transl=torch.from_numpy(smpl_trans[n]).float().to(args.device),
            ).joints.detach().cpu().numpy().reshape(T, -1, 3)[:, 0:24, :]  # only take 24 standard joints
        motion_beats = motion_peak_onehot(keypoints3d)

        # get beat alignment scores
        beat_score = alignment_score(audio_beats, motion_beats, sigma=2.12)
        beat_scores.append(beat_score)


    return sum(beat_scores) / N

def calculate_beat_score_generated(result_file, motion_dir, audio_features_dir, smpl):

    data = pkl.load(open(result_file,"rb"))

    transl = data['transl'] # (N,T, 3)
    body_pose = data['body_pose'] # (N, T, 69)
    global_orient = data['global_orient'] # (N, T, 3)


    N, T = body_pose.shape[:2]
    J = smpl.NUM_JOINTS  # 23


    # get real data music beats
    seq_name = os.path.splitext(os.path.basename(result_file))[0]
    audio_name = seq_name
    #audio_name = "_".join(seq_name.split("_")[:-1])

    if not os.path.exists(os.path.join(audio_features_dir, f"{audio_name}.npy")):
        print(os.path.join(audio_features_dir, f"{audio_name}.npy"), "Not exist!")
        return False
    audio_feature = np.load(os.path.join(audio_features_dir, f"{audio_name}.npy"))

    audio_beats = audio_feature[:T, -1]  # last dim is the music beats

    beat_scores = []

    for n in range(N):
        # keypoints3d = all_joints3d[n]
        with torch.no_grad():
            keypoints3d = smpl.forward(
                global_orient=torch.from_numpy(global_orient[n]).float().to(args.device),
                body_pose=torch.from_numpy(body_pose[n]).float().to(args.device),
                transl=torch.from_numpy(transl[n]).float().to(args.device),
            ).joints.detach().cpu().numpy().reshape(T, -1, 3)[:, 0:24, :]  # only take 24 standard joints
        motion_beats = motion_peak_onehot(keypoints3d)

        # get beat alignment scores
        beat_score = alignment_score(audio_beats, motion_beats, sigma=2.12)
        beat_scores.append(beat_score)


    return sum(beat_scores) / N

def main():

    motion_dir = os.path.join(args.data_dir, "motions_smpl/")

    # set smpl
    smpl = SMPL(model_path=args.smpl_dir, gender='MALE', batch_size=1).to(args.device)

    # create list of sequence names
    seq_names = []
    if "train" in args.split:
        seq_names += np.loadtxt(os.path.join(args.data_dir, "train_split_sequence_names.txt"), dtype=str).tolist()
    if "val" in args.split:
        seq_names += np.loadtxt(os.path.join(args.data_dir, "val_split_sequence_names.txt"), dtype=str).tolist()
    if "test" in args.split:
        seq_names += np.loadtxt(os.path.join(args.data_dir, "test_split_sequence_names.txt"), dtype=str).tolist()
    ignore_list = []
    seq_names = [name for name in seq_names if name not in ignore_list]

    ## calculate score on real data
    # n_samples = len(seq_names)
    # beat_scores = []
    # for i, seq_name in enumerate(seq_names):
    #     print("processing %d / %d" % (i + 1, n_samples))
    #     sample_beat_score = calculate_beat_score_gt(seq_name, motion_dir, args.audio_features_dir, smpl)
    #     beat_scores.append(sample_beat_score)
    # print ("\nBeat score on real data: %.3f\n" % (sum(beat_scores) / n_samples))

    # calculate score on generated motion data
    result_files = sorted(glob.glob(args.result_files))
    n_samples = 0 #len(result_files)
    beat_scores = []
    for result_file in tqdm.tqdm(result_files):
        generated_beat_score = calculate_beat_score_generated(result_file,  motion_dir, args.audio_features_dir, smpl)
        if generated_beat_score:
            beat_scores.append(generated_beat_score)
            n_samples += 1
    print ("\nBeat score on generated data: %.3f\n" % (sum(beat_scores) / n_samples))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, default='datasets/gdance/')
    parser.add_argument('--audio_features_dir', type=str, default='datasets/gdance/librosa_features')
    parser.add_argument('--split', type=str, default='testval', choices=['train', 'testval'])
    parser.add_argument('--result_files', type=str, default='infer_test_out/*.pkl')

    parser.add_argument('--smpl_dir', type=str, default='models_smpl/', help='input local dictionary that stores SMPL data.')
    parser.add_argument('--device',type=str, default='cpu',)
    args = parser.parse_args()



    main()