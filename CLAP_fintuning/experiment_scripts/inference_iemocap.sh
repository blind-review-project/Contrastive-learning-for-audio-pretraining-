# Load necessary modules or environment setups for local machine
# On local machine, you may not need 'module load' commands, but ensure you have CUDA installed and accessible.

# export NCCL_PROTO=simple
# export NCCL_DEBUG=info
# export NCCL_TREE_THRESHOLD=0
# export FI_PROVIDER=efa  # If you have a local EFA setup, otherwise omit#
#export TRANSFORMERS_CACHE=/path/to/your/local/transformers_cache  # Modify this path to point to your local cache

# Activate the virtual environment (change the path to your local virtual environment)
#source /path/to/your/local/virtualenv/bin/activate

# Change directory to the CLAP src folder (modify path to your local setup) #--dataset-type="webdataset" \--remotedata \
cd /data/jiacheng/CLAP/src/laion_clap/
export HF_HOME=~/.cache/huggingface/hub/hub/
# Run the Python script directly without using `srun`
#for lr in 1e-2 1e-3 1e-5 1e-6 1e-7; do
CUDA_VISIBLE_DEVICES=2 python -m evaluate.eval_linear_probe \
        --save-frequency 50 \
        --save-top-performance 3 \
        --save-most-recent \
        --dataset-type="webdataset" \
        --datasetpath="/data/jiacheng/webdata/" \
        --precision="fp32" \
        --warmup 0 \
        --batch-size=128 \
        --lr=1e-4 \
        --wd=0.1 \
        --epochs=0 \
        --workers=4 \
        --use-bn-sync \
        --amodel PANN-14 \
        --tmodel roberta \
        --report-to "wandb" \
        --wandb-notes "11.12-finetune-iemocap-UA" \
        --datasetnames "iemocap" \
        --datasetinfos "train" \
        --seed 3407 \
        --logs /data/jiacheng/CLAP/clap_logs/iemoclap/12.11_emo+meld+mosei_wa\
        --gather-with-grad \
        --lp-loss="ce" \
        --lp-metrics="acc" \
        --lp-lr=1e-4 \
        --lp-mlp \
        --class-label-path="/data/jiacheng/CLAP/class_labels/iemocap_labels.json" \
        --openai-model-cache-dir ~/.cache/huggingface/hub/hub \
        --pretrained="/data/jiacheng/CLAP/laion_clap/630k/" \
        --resume="/data/jiacheng/CLAP/ckpt_eval/lp_pe15_latest_bare.pt"
        --data-filling "repeatpad" \
        --data-truncating "rand_trunc" \
        --optimizer "adam" \
        --freeze-text
# CUDA_VISIBLE_DEVICES=0
# for lr in 1e-2 1e-3 1e-5 1e-6 1e-7; do 
#     python -m evaluate.eval_linear_probe \
#         --save-frequency 50 \
#         --save-top-performance 3 \
#         --save-most-recent \
#         --dataset-type="webdataset" \
#         --datasetpath="/data/jiacheng/webdata/" \
#         --precision="fp32" \
#         --warmup 0 \
#         --batch-size=256 \
#         --lr=$lr \
#         --wd=0.1 \
#         --epochs=300 \
#         --workers=4 \
#         --use-bn-sync \
#         --amodel PANN-14 \
#         --tmodel roberta \
#         --report-to "wandb" \
#         --wandb-notes "10.28-finetune-iemocap-UA-$lr" \
#         --datasetnames "iemocap" \
#         --datasetinfos "train" \
#         --seed 3407 \
#         --logs /data/jiacheng/CLAP/clap_logs/iemoclap/ua_lr_$lr \
#         --gather-with-grad \
#         --lp-loss="ce" \
#         --lp-metrics="ua" \
#         --lp-lr=1e-4 \
#         --lp-mlp \
#         --class-label-path="/data/jiacheng/CLAP/class_labels/iemocap_labels.json" \
#         --openai-model-cache-dir ~/.cache/huggingface/hub/hub \
#         --pretrained="/data/jiacheng/CLAP/laion_clap/630k/" \
#         --data-filling "repeatpad" \
#         --data-truncating "rand_trunc" \
#         --optimizer "adam" \
#         --freeze-text
# done


