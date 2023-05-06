# Train Trainscript
python train_transcript.py \
	--train-data data/opencpop_train.json \
	--dev-data data/opencpop_test_clean.json \
	--whisper-model medium \
	--device cuda \
	--train-batch-size 4 \
	--dev-batch-size 8 \
	--accum-grad-steps 8 \
	--lr 3e-6 \
	--train-steps 1500 \
	--eval-steps 100 \
	--warmup-steps 150 \
	--freeze-encoder \
	--save-dir exp/230506_opencpop_trans_medium_freeze

# Inference
python inference_transcript.py \
	-f data/opencpop_test_clean.json \
	--model-dir exp/230506_opencpop_trans_medium \
	--output exp/230506_opencpop_trans_medium/result.json
# Evaluate
python evaluate_transcript.py -f exp/230506_opencpop_trans_medium/result.json
