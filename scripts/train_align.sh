align_train=${1}
align_dev=${2}

whipser_model=${3}
align_model_dir=exp/230509_${whipser_model}_opencpop_align_2

mkdir ${align_model_dir}
cp ${0} ${align_model_dir}

# Train Alignment
python train_alignment.py \
    --train-data ${align_train} \
    --dev-data ${align_dev} \
    --whisper-model ${whipser_model} \
    --device cuda \
    --train-batch-size 2 \
    --accum-grad-steps 8 \
    --dev-batch-size 8 \
    --lr 1e-3 \
    --train-steps 2500 \
    --eval-steps 100 \
    --warmup-steps 250 \
    --save-dir ${align_model_dir}

# Inference Alignment
python inference_align.py \
    -f ${align_dev} \
    --model-dir ${align_model_dir}