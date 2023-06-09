train_data=${1}
dev_data=${2}

whisper_model=${3}

model_dir=exp/230508_${whipser_model}_opencpop_trans_with_timestamps

mkdir -p ${model_dir}
cp ${0} ${model_dir}

# Train Trainscript
python train_transcript.py \
	--train-data ${train_data} \
	--dev-data ${dev_data} \
	--whisper-model ${whisper_model} \
	--device cuda \
	--train-batch-size 4 \
	--dev-batch-size 8 \
	--accum-grad-steps 4 \
	--lr 5e-6 \
	--train-steps 2500 \
	--eval-steps 100 \
	--warmup-steps 250 \
	--freeze-encoder \
	--save-dir ${model_dir}

# Inference
result_file=${model_dir}/result.json

python inference_transcript.py \
	-f data/opencpop_test_clean.json \
	--get-timestamps \
	--model-dir ${model_dir} \
	--output ${result_file}
# Evaluate
python evaluate_transcript.py \
	-f ${result_file} \
	--evaluate-mae
