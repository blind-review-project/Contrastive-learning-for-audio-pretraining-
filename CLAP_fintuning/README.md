## Datasets
Follow the example of webdata conversion to convert the dataset into webdata. https://github.com/LAION-AI/audio-dataset
- IEMOCAP: https://sail.usc.edu/iemocap/
- MELD: https://www.kaggle.com/datasets/zaber666/meld-dataset
- SAVEE: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
- RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

**After downloading the data, unzip it into the `datasets` folder.**


## Environment Installation
If you want to check and reuse our model in your project instead of directly using the pip library, you need to install the same environment as we use. Please run the following commands:
```bash
conda create -n clap python=3.10
conda activate clap
git clone https://github.com/LAION-AI/CLAP.git
cd CLAP
# you can also install pytorch by following the official instruction (https://pytorch.org/get-started/locally/)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


## How to Finetune
Modify the configuration files under `experiment_scripts` and `class_label` accordingly based on IEMOCAP. Load the corresponding pre-trained model into the appropriate section of the configuration file. Then run one of the following:

```bash
bash experiment_scripts/local_finetune_iemocap.sh
```


## Inference
Inference on IEMOCAP is run with:

```bash
bash inference_iemocap.sh
```

Before running, download the pretrained weights from the [Google Drive folder](https://drive.google.com/drive/folders/1lTUTnDLADJvHZ_KBIRr4Bp4yiP9i8Ro_):

- Encoder base weight: `630k-audioset-best.pt`
- Linear probe weight: `lp_pe15_latest_bare.pt`

Then set the following hyperparameters in the script:

```bash
--pretrained=/path/to/backbone_dir                       # directory containing 630k-audioset-best.pt
--resume=/path/to/lp_pe15_latest_bare.pt                 # linear probe weight
--epochs=0                                               # inference only, no training
```


## Dataset Note
Since CLEP is built on top of the CLAP framework, before attempting any training or inference, please make sure to convert your audio dataset into the WebDataset format required by CLAP. **Otherwise, both CLAP and CLEP models will not be able to properly load or process the data.**


