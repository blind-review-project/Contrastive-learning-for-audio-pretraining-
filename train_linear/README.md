## Datasets
Follow the example of webdata conversion to convert the dataset into webdata. https://github.com/LAION-AI/audio-dataset
Iemocap: https://sail.usc.edu/iemocap/
Meld: https://www.kaggle.com/datasets/zaber666/meld-dataset
Savee: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
Ravdness: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
**After downloading the data, unzip it into the 'datasets' folder.**

## Requirements

**How to Start:**
```shell
conda env create -f environment.yml
```

## Training & Inference 

Here, taking the training and inference of iemocap as an example:
```shell
# Training
cd scripts
sh run_clap_iemocap_ps.sh
```