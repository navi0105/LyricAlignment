model_dir=${1}

opencpop_test=data/opencpop_test_clean.json
opensinger_test=data/opensinger_test_clean.csv
mpop600_test=data/mpop600_test.json
mir1k=data/mir1k.json
mir1k_demucs=data/mir1k_demucs.json


python inference_transcript.py \
    -f ${opencpop_test} \
    --model-dir ${model_dir} \
    --use-pretrained \
    -o ${model_dir}/result_opencpop_test.json
echo "Opencpop Test"
python evaluate_transcript.py -f ${model_dir}/result_opencpop_test.json

python inference_transcript.py \
    -f ${opensinger_test} \
    --model-dir ${model_dir} \
    --use-pretrained \
    -o ${model_dir}/result_opensinger_test.json
echo "OpenSinger Test"
python evaluate_transcript.py -f ${model_dir}/result_opensinger_test.json

python inference_transcript.py \
    -f ${mpop600_test} \
    --model-dir ${model_dir} \
    --use-pretrained \
    -o ${model_dir}/result_mpop600_test.json
echo "MPOP600 Test"
python evaluate_transcript.py -f ${model_dir}/result_mpop600_test.json

## MIR-1k mixture
python inference_transcript.py \
    -f ${mir1k} \
    --model-dir ${model_dir} \
    --use-pretrained \
    --is-mir1k 1 \
    -o ${model_dir}/result_mir1k_mixture.json
echo "MIR-1k Mixture"
python evaluate_transcript.py -f ${model_dir}/result_mir1k_mixture.json

## MIR-1k voice
python inference_transcript.py \
    -f ${mir1k} \
    --model-dir ${model_dir} \
    --use-pretrained \
    --is-mir1k 2 \
    -o ${model_dir}/result_mir1k_voice.json
echo "MIR-1k Voice"
python evaluate_transcript.py -f ${model_dir}/result_mir1k_voice.json

## MIR-1k demucs
python inference_transcript.py \
    -f ${mir1k_demucs} \
    --model-dir ${model_dir} \
    --use-pretrained \
    -o ${model_dir}/result_mir1k_demucs.json
echo "MIR-1k demucs"
python evaluate_transcript.py -f ${model_dir}/result_mir1k_demucs.json