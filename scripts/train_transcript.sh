transcript_train=${1}
transcript_dev=${2}

multitask_train=${3}
multitask_dev=${4}

whipser_model=${5}

transcript_model_dir=${6}

mkdir -p ${transcript_model_dir}
cp ${0} ${transcript_model_dir}

# Train
python train_multitask.py \
    --train-data ${multitask_train} ${transcript_train} \
    --dev-data ${multitask_dev} ${transcript_dev} \
    --whisper-model ${whipser_model} \
    --device cuda:1 \
    --train-batch-size 2 \
    --dev-batch-size 8 \
    --accum-grad-steps 8 \
    --use-ctc-loss \
    --lr 3e-5 \
    --train-steps 2000 \
    --eval-steps 100 \
    --warmup-steps 200 \
    --save-dir ${transcript_model_dir} | tee ${transcript_model_dir}/log.txt || exit 1;

bash scripts/evaluate_benchmark.sh ${transcript_model_dir} true | tee ${transcript_model_dir}/evaluate_log.txt || exit 1;