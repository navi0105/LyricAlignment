multitask_dev=${1}

transcript_dev=${2}

multitask_model_dir=${3}

# Inference Alignment
echo "Inference Alignment"
python inference_align.py \
    -f ${multitask_dev} \
    --model-dir ${multitask_model_dir} \
    --use-pypinyin

# Inference Transcript
echo "Inference Transcription"
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
python evaluate_transcript.py \
    -f ${result_1}

python evaluate_transcript.py \
    -f ${result_2}