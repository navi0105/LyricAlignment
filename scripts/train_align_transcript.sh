align_train=${1}
align_dev=${2}
transcript_train=${3}
transcript_dev=${4}

whipser_model=${5}
align_model_dir=exp/230508_${whipser_model}_opencpop_align
transcript_model_dir=${align_model_dir}_opensinger_trans_freeze_notime

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
    --lr 3e-3 \
    --train-steps 2000 \
    --eval-steps 100 \
    --warmup-steps 150 \
    --save-dir ${align_model_dir}

mkdir ${transcript_model_dir}
cp ${0} ${transcript_model_dir}

# Train Transcript
python train_transcript.py \
    --train-data ${transcript_train} \
    --dev-data ${transcript_dev} \
    --align-model-dir ${align_model_dir} \
    --device cuda \
    --train-batch-size 2 \
    --accum-grad-steps 8 \
    --dev-batch-size 8 \
    --lr 5e-6 \
    --train-steps 1500 \
    --eval-steps 100 \
    --warmup-steps 150 \
    --no-timestamps \
    --freeze-encoder \
    --save-dir ${transcript_model_dir}

result_1=${transcript_model_dir}/result_opensinger_test.json
result_2=${transcript_model_dir}/result_opencpop_test.json

# Inference Alignment
python inference_align.py \
    -f ${align_dev} \
    --model-dir ${transcript_model_dir}

python inference_transcript.py \
    -f ${transcript_dev} \
    --model-dir ${transcript_model_dir} \
    --device cuda \
    --output ${result_1}

python inference_transcript.py \
    -f ${align_dev} \
    --model-dir ${transcript_model_dir} \
    --device cuda \
    --output ${result_2}

# Evaluate
python evaluate_transcript.py \
    -f ${result_1}

python evaluate_transcript.py \
    -f ${result_2}