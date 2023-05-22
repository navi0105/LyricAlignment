model_dir=${1}

python inference_transcript.py -f data/mir1k.json --model-dir ${model_dir} -o ${model_dir}/result_mir1k.json
python inference_transcript.py -f data/mpop600_test.json --model-dir ${model_dir} -o ${model_dir}/result_mpop600_test.json
python inference_transcript.py -f data/opensinger_test_clean.csv --model-dir ${model_dir} -o ${model_dir}/result_opensinger_test.json

echo "MIR-1k"
python evaluate_transcript.py -f ${model_dir}/result_mir1k.json
echo "MPOP600"
python evaluate_transcript.py -f ${model_dir}/result_mpop600_test.json
echo "OpenSinger Test"
python evaluate_transcript.py -f ${model_dir}/result_opensinger_test.json