# Lyric Alignment
This is a source code of paper "Adapting pretrained speech model for Mandarin lyrics transcription and alignment", you can use this repository to reproduce the experiment in our paper.


## Create Conda environment & Install package
```bash
pip install -r requirements.txt 
```

## Usage
### Data Preparation
Go to `dataset_preprocessing/` to get more information of data preparation.

### Train & evaluate by script
After you finish data preprocessing, you can run our sample training script `scripts/train_multitask.sh` to train & evaluate the model.
```bash
bash scripts/train_multitask.sh [train_data] [dev_data] [test_data] [model_dir]
```
I have also set some training argument in the script, these arguments are the final argument we used in our experiment, you can modify them to get different result. 

### Training
Excute `train_multitask.py` to train model, don't forget preprocess your data first.
You can set `--train-alignment` / `--train-transcript` batch to determine which task you want to train. Of course you can set two flags at the same time for multitask training.
```bash
python train_multitask.py \
    --train-data [train_data_1] [train_data_2] ... [train_data_N] \
    --dev-data [dev_data_1] [dev_data_2] ... [dev_data_N] \
    --whisper-model [whisper_pretrained_model] \
    --train-alignment \
    --train-transcript \
    --use-ctc-loss \
    --save-dir [save_dir]
```
### Inference & Evaluate
After training finished, you can execute our inference code to evaluate the model performance.
```bash
# Alignment Evaluate
# set `--predict-sil` flag activate if you have used CTC Loss during traing phase.
python inference_alignment.py \
    -f [test_data] \
    --model-dir [model_dir] \
    --predict-sil  \

# Transcript Evaluate
# Get transcript result first
python inference_transcript.py \
    -f [test_data] \
    --model-dir [model_dir] \
    --output [output_file_path]
# Evaluate the inference result file
python evaluate_transcript.py \
    -f [result_file_path]
```

## TODO
### README
- [ ] Repo Description
- [x] Data Preprocess
- [ ] Training
- [ ] Evaluate

### Code
- [x] Data Processer
- [x] Dataset & DataLoader
- [x] Module
- [x] Training Code
- [x] Inference Code
- [x] Evaluate Code