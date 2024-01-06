

# [Controllable Group Choreography using Contrastive Diffusion (SIGGRAPH ASIA 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Le_Music-Driven_Group_Choreography_CVPR_2023_paper.pdf)
### *[Nhat Le](https://minhnhatvt.github.io/), [Tuong Do](https://scholar.google.com/citations?user=qCcSKkMAAAAJ&hl=en), [Khoa Do](https://aioz-ai.github.io/GCD/), [Hien Nguyen](https://aioz-ai.github.io/GCD/), [Erman Tjiputra](https://sg.linkedin.com/in/erman-tjiputra), [Quang D. Tran](https://scholar.google.com/citations?user=DbAThEgAAAAJ&hl=en), [Anh Nguyen](https://cgi.csc.liv.ac.uk/~anguyen/)*
### [[Project Page](https://aioz-ai.github.io/GCD/)] [[Paper](https://dl.acm.org/doi/abs/10.1145/3618356)] [[Arxiv](https://arxiv.org/abs/2310.18986)]



![](https://aioz-ai.github.io/GCD/static/figures/Intro.png)*<center>  We present a contrastive diffusion method that can control the consistency and diversity in group choreography </center>*


## Table of Contents
1. [Prerequisites](#prerequisites)


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

Our model also operates on input music features to generate corresponding motions. We provide pre-extracted (Jukebox) features from GDANCE music sequences in [here](). 

If you want to extract the music features yourself, please check out [this directory]().

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





<!-- ## Citation
```
@inproceedings{aiozGdance,
    author    = {Le, Nhat and Pham, Thang and Do, Tuong and Tjiputra, Erman and Tran, Quang D. and Nguyen, Anh},
    title     = {Music-Driven Group Choreography},
    journal   = {CVPR},
    year      = {2023},
}		
``` -->

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


