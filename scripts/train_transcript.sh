transcript_train=${1}
transcript_dev=${2}

align_model_dir=${3}

transcript_model_dir=${align_model_dir}_opensinger_trans_freeze

mkdir -p ${transcript_model_dir}
cp ${0} ${transcript_model_dir}

# Train
python train_transcript.py \
    --train-data ${transcript_train} \
    --dev-data ${transcript_dev} \
    --align-model-dir ${align_model_dir} \
    --train-batch-size 4 \
    --dev-batch-size 8 \
    --accum-grad-steps 4 \
    --device cuda \
    --no-timestamps \
    --freeze-encoder \
    --lr 1e-5 \
    --train-steps 1500 \
    --eval-steps 100 \
    --warmup-steps 150 \
    --save-dir ${transcript_model_dir}

# Inference
result_file=${transcript_model_dir}/result_opensinger_test.json

python inference_transcript.py \
    -f ${transcript_dev} \
    --model-dir ${transcript_model_dir} \
    --device cuda \
    --output ${result_file}

python evaluate_transcript.py \
    -f ${result_file}