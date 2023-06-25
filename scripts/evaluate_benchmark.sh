model_dir=${1}
use_ctc=${2}

device=${3}

opencpop_test=data/opencpop_test_clean.json
opensinger_test=data/opensinger_test_clean_0511.csv
opensinger_test_demucs=data/opensinger_test_demucs_clean_0511.csv
mpop600_test=data/mpop600_test_clean.json
mir1k=data/mir1k_clean.json
mir1k_demucs=data/mir1k_demucs_clean.json

# Alignment
echo "Opencpop Test Alignment"
python inference_align.py \
    -f ${opencpop_test} \
    --model-dir ${model_dir} \
    --use-pypinyin \
    --predict-sil ${use_ctc} \
    --device ${device}

echo "MPOP600 Test Alignment"
python inference_align.py \
    -f ${mpop600_test} \
    --model-dir ${model_dir} \
    --use-pypinyin \
    --predict-sil ${use_ctc} \
    --device ${device}

echo "MIR-1k Demucs Alignment"
python inference_align.py \
    -f data/mir1k_demucs_clean_align_only.json \
    --model-dir ${model_dir} \
    --use-pypinyin \
    --predict-sil ${use_ctc} \
    --device ${device}

# Transcript
python inference_transcript.py \
    -f ${opencpop_test} \
    --model-dir ${model_dir} \
    -o ${model_dir}/result_opencpop_test.json \
    --device ${device}
echo "Opencpop Test"
python utils/postprocess.py -f ${model_dir}/result_opencpop_test.json
python evaluate_transcript.py -f ${model_dir}/result_opencpop_test.json

python inference_transcript.py \
    -f ${opensinger_test} \
    --model-dir ${model_dir} \
    -o ${model_dir}/result_opensinger_test.json \
    --device ${device}
echo "OpenSinger Test"
python utils/postprocess.py -f ${model_dir}/result_opensinger_test.json
python evaluate_transcript.py -f ${model_dir}/result_opensinger_test.json

python inference_transcript.py \
    -f ${opensinger_test_demucs} \
    --model-dir ${model_dir} \
    -o ${model_dir}/result_opensinger_test_demucs.json \
    --device ${device}
echo "OpenSinger Test Demucs"
python utils/postprocess.py -f ${model_dir}/result_opensinger_test_demucs.json
python evaluate_transcript.py -f ${model_dir}/result_opensinger_test_demucs.json

python inference_transcript.py \
    -f ${mpop600_test} \
    --model-dir ${model_dir} \
    -o ${model_dir}/result_mpop600_test.json \
    --device ${device}
echo "MPOP600 Test"
python utils/postprocess.py -f ${model_dir}/result_mpop600_test.json
python evaluate_transcript.py -f ${model_dir}/result_mpop600_test.json

## MIR-1k mixture
python inference_transcript.py \
    -f ${mir1k} \
    --model-dir ${model_dir} \
    --is-mir1k 1 \
    -o ${model_dir}/result_mir1k_mixture.json \
    --device ${device}
echo "MIR-1k Mixture"
python utils/postprocess.py -f ${model_dir}/result_mir1k_mixture.json
python evaluate_transcript.py -f ${model_dir}/result_mir1k_mixture.json

## MIR-1k voice
python inference_transcript.py \
    -f ${mir1k} \
    --model-dir ${model_dir} \
    --is-mir1k 2 \
    -o ${model_dir}/result_mir1k_voice.json \
    --device ${device}
echo "MIR-1k Voice"
python utils/postprocess.py -f ${model_dir}/result_mir1k_voice.json
python evaluate_transcript.py -f ${model_dir}/result_mir1k_voice.json

## MIR-1k demucs
python inference_transcript.py \
    -f ${mir1k_demucs} \
    --model-dir ${model_dir} \
    -o ${model_dir}/result_mir1k_demucs.json \
    --device ${device}
echo "MIR-1k demucs"
python utils/postprocess.py -f ${model_dir}/result_mir1k_demucs.json
python evaluate_transcript.py -f ${model_dir}/result_mir1k_demucs.json





