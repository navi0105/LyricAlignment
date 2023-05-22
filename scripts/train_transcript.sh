transcript_train=${1}
transcript_dev=${2}

multitask_train=${3}
multitask_dev=${4}

align_model_dir=${5}

transcript_model_dir=${align_model_dir}_trans_freeze

mkdir -p ${transcript_model_dir}
cp ${0} ${transcript_model_dir}

# Train
python train_transcript.py \
    --train-data ${transcript_train} ${multitask_train} \
    --dev-data ${transcript_dev} ${multitask_dev} \
    --align-model-dir ${align_model_dir} \
    --train-batch-size 2 \
    --dev-batch-size 8 \
    --accum-grad-steps 8 \
    --device cuda \
    --no-timestamps \
    --freeze-encoder \
    --lr 5e-4 \
    --train-steps 1500 \
    --eval-steps 100 \
    --warmup-steps 150 \
    --save-dir ${transcript_model_dir} | tee ${transcript_model_dir}/log.txt

# Inference
bash scripts/evaluate_benchmark.sh ${transcript_model_dir} true | tee -a ${transcript_model_dir}/log.txt