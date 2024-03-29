

# Controllable Group Choreography using Contrastive Diffusion (SIGGRAPH ASIA 2023)
### *[Nhat Le](https://minhnhatvt.github.io/), [Tuong Do](https://scholar.google.com/citations?user=qCcSKkMAAAAJ&hl=en), [Khoa Do](https://aioz-ai.github.io/GCD/), [Hien Nguyen](https://aioz-ai.github.io/GCD/), [Erman Tjiputra](https://sg.linkedin.com/in/erman-tjiputra), [Quang D. Tran](https://scholar.google.com/citations?user=DbAThEgAAAAJ&hl=en), [Anh Nguyen](https://cgi.csc.liv.ac.uk/~anguyen/)*
### [[Project Page](https://aioz-ai.github.io/GCD/)] [[Paper](https://dl.acm.org/doi/abs/10.1145/3618356)] [[Arxiv](https://arxiv.org/abs/2310.18986)]



![](https://aioz-ai.github.io/GCD/static/figures/Intro.png)*<center> We present a contrastive diffusion method that can control the consistency and diversity in group choreography </center>*


## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Training](#training-gcd-from-scratch)


## Prerequisites

This code was tested on `Ubuntu 18.04.5 LTS` and requires:

* Python 3.7 & 3.8
* anaconda3 (or miniconda3)
* CUDA supported GPU 

### 1. Environment Setup
First of all, please setup the virtual environment with Anaconda:
```bash
conda create -n gcd python=3.8
conda activate gcd
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

<!-- ### Body Model

Our code relies on [SMPL](https://smpl.is.tue.mpg.de/) as the body model. You can download our processed version from [here](). -->

### 2. Data

We use [GDANCE](https://github.com/aioz-ai/AIOZ-GDANCE) dataset to train and evaluate models in our experiments. Please [download](https://vision.aioz.io/f/430eb9d90552480e8b4e/?dl=1) and extract the data into `./datasets/` (Otherwise you may need to update the running option to point to the path you have extracted).

Our model also operates on input music features to generate corresponding motions. We provide pre-extracted (Jukebox) features from GDANCE music sequences in [here](https://huggingface.co/aiozai/JukeBoxFeatures/resolve/main/jukebox_features.zip). 

If you want to extract the music features yourself, please check out [this repository](https://github.com/aioz-ai/GCD/tree/main/extract).

The data directory structure should look like this:
```bash
.
└── datasets/gdance/
    ├── motions_smpl/
    │   ├── sequence_name.pkl
    │   └── ...
    ├── music/
    │   ├── sequence_name.wav
    │   └── ...
    ├── jukebox_features/
    │   ├── sequence_name.npy
    │   └── ...
    ├── train_split_sequence_names.txt
    ├── val_split_sequence_names.txt
    └── test_split_sequence_names.txt
```


## Training GCD from scratch
> Note: you may run `accelerate config` before starting in order to configure training with multiple GPUs or mixed precision (we use fp16).



```bash
accelerate launch --mixed_precision fp16 -m train.train --save_dir save_ckpt/gcd --datapath "datasets/gdance" --split_file "train_split_sequence_names.txt" --music_extra_token --target_seq_len 150 --max_persons 5 --layers 5 --cond_mask_prob 0.2 --lambda_main 1.0 --lambda_vel 1.0 --lambda_rcxyz 1.0 --lambda_fc 5.0  --lambda_contrastive 0.00001 --use_film --use_style --disc_num_layers 2 --disc_time_aware --intra_replace_prob 0.5 --num_negs 8 --batch_size 32 --num_workers 8 --overwrite --optim Adan --lr 1e-4 --weight_decay 0.02 --num_steps 1000000 --log_interval 20 --save_interval 10000 --resume_checkpoint ""
```


Some useful parameters:
* `--target_seq_len` the fixed sequence length for batching samples for training
* `--max_persons` maximum number of dancers considered per sample
* `--layers` number of layers of the denoising network (transformer decoder blocks)
* `--disc_num_layers`: number of layers of the contrastive encoder
* `--lambda_main, --lambda_vel, ...` corresponding loss weight coefficients
* `--intra_replace_prob` probability for a dancer to be replaced by another in negative samples
* `--num_negs` number of negative samples for contrastive learning
* `--num_steps` number of expected training iterations
* `--save_interval` interval to save model checkpoint (every specified iteration)
* `--resume_checkpoint` path to model checkpoint to resume previous training session (we also save the model with currently highest iterations in `last.ckpt`)

We also support logging and monitoring with [wandb](https://wandb.ai/site), where the training progress will be logged and visualized onto their web interface. You can specify the project name `--wandb_pj_name`  and experiment name `--exp_name`.






## Citation

If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:

```
@article{le2023controllable,
  title={Controllable Group Choreography Using Contrastive Diffusion},
  author={Le, Nhat and Do, Tuong and Do, Khoa and Nguyen, Hien and Tjiputra, Erman and Tran, Quang D and Nguyen, Anh},
  journal={ACM Transactions on Graphics (TOG)},
  volume={42},
  number={6},
  pages={1--14},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```

If you find that the AIOZ-GDance dataset is useful, you could cite the following paper:
```
@inproceedings{aiozGdance,
    author    = {Le, Nhat and Pham, Thang and Do, Tuong and Tjiputra, Erman and Tran, Quang D. and Nguyen, Anh},
    title     = {Music-Driven Group Choreography},
    journal   = {CVPR},
    year      = {2023},
}		
```

## License
Software Copyright License for non-commercial scientific research purposes.
Please read carefully the following [terms and conditions](LICENSE) and any accompanying
documentation before you download and/or use AIOZ-GDANCE data, model and
software, (the "Data & Software"), including 3D meshes, images, videos,
textures, software, scripts, and animations. By downloading and/or using the
Data & Software (including downloading, cloning, installing, and any other use
of the corresponding github repository), you acknowledge that you have read
these [terms and conditions](LICENSE), understand them, and agree to be bound by them. If
you do not agree with these [terms and conditions](LICENSE), you must not download and/or
use the Data & Software. Any infringement of the terms of this agreement will
automatically terminate your rights under this [License](LICENSE).


## Acknowledgement


