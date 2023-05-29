model_dir=${1}

opencpop_test=data/opencpop_test_clean.json
opensinger_test=data/opensinger_test_clean.csv
mpop600_test=data/mpop600_test.json
mir1k=data/mir1k.json
mir1k_demucs=data/mir1k_demucs.json

# Opencpop
echo "Opencpop Test"
python inference_transcript_encoder.py \
    -f ${opencpop_test} \
    --model-dir ${model_dir} \
    --output ${model_dir}/result_opencpop_test_encoder.json

python evaluate_transcript.py \
    -f ${model_dir}/result_opencpop_test_encoder.json

# OpenSinger
echo "OpenSinger Test"
python inference_transcript_encoder.py \
    -f ${opensinger_test} \
    --model-dir ${model_dir} \
    --output ${model_dir}/result_opensinger_test_encoder.json

python evaluate_transcript.py \
    -f ${model_dir}/result_opensinger_test_encoder.json

# MPOP600
echo "MPOP600 Test"
python inference_transcript_encoder.py \
    -f ${mpop600_test} \
    --model-dir ${model_dir} \
    --output ${model_dir}/result_mpop600_test_encoder.json

python evaluate_transcript.py \
    -f ${model_dir}/result_mpop600_test_encoder.json

# MIR-1k
echo "MIR-1k mixture"
python inference_transcript_encoder.py \
    -f ${mir1k} \
    --model-dir ${model_dir} \
    --is-mir1k 1 \
    --output ${model_dir}/result_mir1k_mixture_encoder.json

python evaluate_transcript.py \
    -f ${model_dir}/result_mir1k_mixture_encoder.json

echo "MIR-1k voice"
python inference_transcript_encoder.py \
    -f ${mir1k} \
    --model-dir ${model_dir} \
    --is-mir1k 2 \
    --output ${model_dir}/result_mir1k_voice_encoder.json

python evaluate_transcript.py \
    -f ${model_dir}/result_mir1k_voice_encoder.json

echo "MIR-1k Demucs"
python inference_transcript_encoder.py \
    -f ${mir1k_demucs} \
    --model-dir ${model_dir} \
    --output ${model_dir}/result_mir1k_demucs_encoder.json

python evaluate_transcript.py \
    -f ${model_dir}/result_mir1k_demucs_encoder.json