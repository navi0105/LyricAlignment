# You may use this script for multitask training after you have finished data preprocess.
train_data=${1}
dev_data=${2}
test_data=${3}
model_dir=${4}
training_setting=${5}

if [ ${training_setting} = "alignment" ]
then
    whisper_model='medium'
    device='cuda'
    train_batch_size=2
    dev_batch_size=8
    accum_grad_steps=8
    lr=0.005
    backbone_lr=5e-6
    train_steps=2000
    eval_steps=200
    warmup_steps=200
    seed=114514
else
    whisper_model='medium'
    device='cuda'
    train_batch_size=2
    dev_batch_size=8
    accum_grad_steps=8
    lr=0.005
    backbone_lr=1e-6
    train_steps=600
    eval_steps=200
    warmup_steps=200
    seed=114514
fi


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
    --backbone-lr ${backbone_lr} \
    --train-steps ${train_steps} \
    --eval-steps ${eval_steps} \
    --warmup-steps ${warmup_steps} \
    --save-dir ${model_dir} \
    --device ${device} \
    --seed ${seed}

# Evaluation
## Alignment
python inference_alignment.py \
    --test-data ${test_data} \
    --model-dir ${model_dir} \
    --use-ctc-loss \
    --device ${device}

## Transcript
python inference_transcript.py \
    --test-data ${test_data} \
    --model-dir ${model_dir} \
    --device ${device} \
    --output ${model_dir}/transcript_result.json \
    --use-groundtruth

python evaluate_transcript.py \
    -f ${model_dir}/transcript_result.json \