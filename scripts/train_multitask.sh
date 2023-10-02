# You may use this script for multitask training after you have finished data preprocess.
train_data=${1}
dev_data=${2}
test_data=${3}
model_dir=${4}

whisper_model='medium'
device='cuda:1'
train_batch_size=2
dev_batch_size=8
accum_grad_steps=8
lr=0.005
train_steps=2000
eval_steps=200
warmup_steps=200

# Training
python train_multitask.py \
    --train-data ${train_data} \
    --dev-data ${dev_data} \
    --whisper-model ${whisper_model} \
    --train-batch-size ${train_batch_size} \
    --dev-batch-size ${dev_batch_size} \
    --accum-grad-steps ${accum_grad_steps} \
    --train-alignment \
    --train-transcript \
    --use-ctc-loss \
    --lr ${lr} \
    --train-steps ${train_steps} \
    --eval-steps ${eval_steps} \
    --warmup-steps ${warmup_steps} \
    --save-dir ${model_dir} \
    --device ${device} \

# Evaluation
## Alignment
python inference_alignment.py \
    --test-data ${test_data} \
    --model-dir ${model_dir} \
    --predict-sil
    --device ${device} \

## Transcript
python inference_transcript.py \
    --test-data ${test_data} \
    --model-dir ${model_dir} \
    --device ${device} \
    --output ${model_dir}/transcript_result.json \

python evaluate_transcript.py \
    -f ${model_dir}/transcript_result.json \