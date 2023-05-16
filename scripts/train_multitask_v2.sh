multitask_train=${1}
multitask_dev=${2}

transcript_train=${3}
transcript_dev=${4}

whipser_model=${5}

multitask_model_dir=${6}

mkdir -p ${multitask_model_dir}
cp ${0} ${multitask_model_dir}

# Train multitask
python train_multitask_v2.py \
    --train-data ${multitask_train} \
    --dev-data ${multitask_dev} \
    --whisper-model ${whipser_model} \
    --device cuda \
    --train-batch-size 2 \
    --dev-batch-size 8 \
    --accum-grad-steps 8 \
    --lr 5e-3 \
    --train-steps 4000 \
    --eval-steps 100 \
    --warmup-steps 400 \
    --save-dir ${multitask_model_dir}

# Inference Transcript
result_1=${multitask_model_dir}/result_opencpop_test.json
result_2=${multitask_model_dir}/result_opensinger_test.json

python inference_transcript.py \
    -f ${multitask_dev} \
    --model-dir ${multitask_model_dir} \
    --device cuda \
    --output ${result_1}

python inference_transcript.py \
    -f ${transcript_dev} \
    --model-dir ${multitask_model_dir} \
    --device cuda \
    --output ${result_2}

# Evaluate
python inference_align.py \
    -f ${multitask_dev} \
    --model-dir ${multitask_model_dir} \
    --use-pypinyin

python evaluate_transcript.py \
    -f ${result_1}

python evaluate_transcript.py \
    -f ${result_2}