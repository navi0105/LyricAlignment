# Train Trainscript
python train_transcript.py \
	--train-data data/opencpop_train.json \
	--dev-data data/opencpop_test_clean.json \
	--whisper-model medium \
	--device cuda \
	--train-batch-size 4 \
	--dev-batch-size 8 \
	--accum-grad-steps 4 \
	--lr 3e-6 \
	--train-steps 1500 \
	--eval-steps 100 \
	--warmup-steps 150 \
	--freeze-encoder \
	--save-dir exp/230507_opencpop_trans_medium

# Inference
python inference_transcript.py \
	-f data/opencpop_test_clean.json \
	--get-timestamps \
	--model-dir exp/230507_opencpop_trans_medium \
	--output exp/230507_opencpop_trans_medium/result.json
# Evaluate
python evaluate_transcript.py -f exp/230506_opencpop_trans_medium/result.json
