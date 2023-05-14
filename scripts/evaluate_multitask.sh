multitask_train=${1}
multitask_dev=${2}

transcript_train=${3}
transcript_dev=${4}

multitask_model_dir=${5}

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
    --model-dir ${multitask_model_dir}

python evaluate_transcript.py \
    -f ${result_1}

python evaluate_transcript.py \
    -f ${result_2}