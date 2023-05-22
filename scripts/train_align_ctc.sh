align_train=${1}
align_dev=${2}

whipser_model=${3}
align_model_dir=${4}

# Train Alignment
python train_alignment_ctc.py \
    --train-data ${align_train} \
    --dev-data ${align_dev} \
    --whisper-model ${whipser_model} \
    --device cuda \
    --train-batch-size 2 \
    --accum-grad-steps 8 \
    --dev-batch-size 8 \
    --lr 5e-3 \
    --train-steps 1000 \
    --eval-steps 100 \
    --warmup-steps 100 \
    --save-dir ${align_model_dir}

# Inference Alignment
python inference_align.py \
    -f ${align_dev} \
    --model-dir ${align_model_dir} \
    --predict_sil