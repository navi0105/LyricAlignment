multitask_train=${1}
multitask_dev=${2}

transcript_train=${3}
transcript_dev=${4}

whipser_model=${5}

multitask_model_dir=${6}

mkdir -p ${multitask_model_dir}
cp ${0} ${multitask_model_dir}

# Train multitask
python train_multitask_v3.py \
    --train-data ${multitask_train} \
    --dev-data ${multitask_dev} \
    --whisper-model ${whipser_model} \
    --device cuda \
    --train-batch-size 2 \
    --dev-batch-size 8 \
    --accum-grad-steps 8 \
    --lr 1e-3 \
    --train-steps 2000 \
    --eval-steps 100 \
    --warmup-steps 200 \
    --save-all-checkpoints true \
    --save-dir ${multitask_model_dir} | tee ${multitask_model_dir}/log.txt

bash scripts/evaluate_benchmark.sh ${multitask_model_dir} false | tee -a ${multitask_model_dir}/log.txt