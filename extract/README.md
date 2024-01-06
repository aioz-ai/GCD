
This repository contains code for extracting music features from [OpenAI's Jukebox](https://openai.com/research/jukebox) model. It requires a single GPU with at least 16GB memory to run. 




## Installlation

```bash
conda create -n jukemir python=3.8
conda activate jukemir

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
pip install fire==0.1.3 tqdm>=4.45.0 soundfile==0.10.3.post1 unidecode==1.1.1 numba==0.48.0 librosa==0.7.2
pip install wget accelerate 
pip install importlib_metadata
pip install setuptools==59.5.0
pip install numpy==1.21.5
python -m pip install --no-cache-dir -e jukebox
```

## Download pretrained weights
```bash
mkdir -p pretrained/5b

wget https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar -P pretrained/5b

wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar -P pretrained/5b
```



## Run extraction
It is highly recommended to pre-compute the representations in advance, since it takes very long. 

To extract the features from a directory containing multiple **.wav** files, run the following command:
```bash
python representation.py --input_dir=<input_dir> --output_dir=<output_dir>
```


## Acknowledgement
We want to thank the contributors of jukemir project for the simplified and comprehensible extraction code of Jukebox model: [jukemir](https://github.com/p-lambda/jukemir), [simplified-jukemir](https://github.com/ldzhangyx/simplified-jukemir).

