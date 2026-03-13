## Datasets
Follow the example of webdata conversion to convert the dataset into webdata. https://github.com/LAION-AI/audio-dataset
Iemocap: https://sail.usc.edu/iemocap/
Meld: https://www.kaggle.com/datasets/zaber666/meld-dataset
Savee: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
Ravdness: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
**After downloading the data, unzip it into the 'datasets' folder.**


## Environment Installation
If you want to check and reuse our model into your project instead of directly using the pip library, you need to install the same environment as we use, please run the following command:
```bash
conda create env -n clap python=3.10
conda activate clap
git clone https://github.com/LAION-AI/CLAP.git
cd CLAP
# you can also install pytorch by following the official instruction (https://pytorch.org/get-started/locally/)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
# How to fientune 
Modify the configuration files under experiment_scripts and class_label accordingly based on IEMOCAP. Load the corresponding pre-trained model into the appropriate section of the configuration file. Run the following command: bash experiment_scripts/local_finetune_iemocap.py or experiment_scripts/local_train_iemocap.py