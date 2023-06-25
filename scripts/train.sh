model_dir=${1}
device=${2}

{
    mkdir -p ${model_dir}
    cp ${0} ${model_dir}

    python train_multitask.py \
        --train-data data/opencpop_train_split.json \
                    data/opencpop_train_aug_snr0_split.json \
                    data/opencpop_train_aug_snr_minus5_split.json \
                    data/opencpop_train_aug_snr_minus10_split.json \
                    data/opensinger_train_split.csv \
                    data/opensinger_train_aug_snr_0_demucs_split.csv \
                    data/opensinger_train_aug_snr_minus5_demucs_split.csv \
                    data/opensinger_train_aug_snr_minus10_demucs_split.csv \
        --dev-data data/opencpop_dev_split.json \
                    data/opencpop_dev_aug_snr0_split.json \
                    data/opencpop_dev_aug_snr_minus5_split.json \
                    data/opencpop_dev_aug_snr_minus10_split.json \
                    data/opensinger_dev_split.csv \
                    data/opensinger_dev_aug_snr_0_demucs_split.csv \
                    data/opensinger_dev_aug_snr_minus5_demucs_split.csv \
                    data/opensinger_dev_aug_snr_minus10_demucs_split.csv \
        --whisper-model medium \
        --device ${device} \
        --train-batch-size 1 \
        --dev-batch-size 8 \
        --accum-grad-steps 16 \
        --use-ctc-loss \
        --lr 5e-3 \
        --train-steps 2000 \
        --eval-steps 100 \
        --warmup-steps 2000 \
        --save-dir ${model_dir} | tee ${model_dir}/log.txt

    bash scripts/evaluate_benchmark.sh ${model_dir} true ${device} | tee ${model_dir}/evaluate_log.txt
    exit
}