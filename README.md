# Lyric Alignment

This is the official source code of the paper:

Jun-You Wang, Chon-In Leong, Yu-Chen Lin, Li Su and Jyh-Shing Roger Jang, "Adapting pretrained speech model for Mandarin lyrics transcription and alignment," *ASRU 2023*.

You can use this repository to reproduce the experiments in our paper.

This repo also includes the character boundary annotation of a subset of the MIR-1k dataset (see `dataset_preprocessing/`).

## Install Python Packages

```bash
pip install -r requirements.txt 
```

## Usage

### Data Preparation

Go to `dataset_preprocessing/` to get more information of data preparation.

### Building the pronunciation lookup table

In the lyrics alignment task, we train the model to predict the framewise probability of syllable pronunciation classes instead of character classes.

In practice, we use the `bert-base-chinese` tokenizer to provide the vocabulary of Mandarin characters. Then, we use pypinyin package ([GitHub - mozillazg/python-pinyin: 汉字转拼音(pypinyin)](https://github.com/mozillazg/python-pinyin)) to determine the pronunciation of these Mandarin characters. We build a lookup table that helps us convert between `bert-base-chinese` token ids and syllable pronunciation classes.

The resulted lookup tables are stored in `bert_base_chinese_pronunce_table.json`. FYI, this work was done by simply running `python get_pronunce_table.py`. 

### Train & evaluate all in one script

After data preprocessing, you can run our sample script `scripts/train_multitask.sh` to train & evaluate the model with the default training arguments.

```bash
bash scripts/train_multitask.sh [train_data] [dev_data] [test_data]\
 [model_dir] [training_setting]
```

where:

`train_data`: Path to the training datasets (json files).

`dev_data`: Path to the dev (validation) datasets (json files).

`test_data`: Path to the test dataset (json file). The test dataset is only used at the evaluation phase.

`model_dir`: The directory to store the model checkpoints and hyperparameters. Need at least 10GB of disk space to store them!

`training_setting`: Choose the preferred training setting. "alignment" will lead to the hyperparameters for lyrics alignment, while "transcription" will lead to the hyperparameters for lyrics transcription. For more details, please refer to our paper.

### Training

Run `train_multitask.py` to train model with customized hyperparameters.

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

where:

`train_data`: Path to the training datasets (json files). Can include multiple datasets. In our experiments, we use 4 versions of the Opencpop dataset (original version and the augmented versions w/ SNR=0, -5, -10) for training.

`dev_data`: Path to the dev (validation) datasets (json files). Can include multiple datasets.

`whisper_pretrained_model`: Specify which pretrained Whisper model to be used as the backbone model. In our experiments, we use Whisper medium (`medium`) as our GPU memory is not very large. If the GPU memory is large enough, perhaps you can try `large`.

`--train-alignment` / `--train-transcript`: Specify which task(s) you want to train on. In our experiments, both flags are set, i.e., both lyrics alignment and lyrics transcription losses are activated.

`--use-ctc-loss`: Set this flag to use the CTC loss for lyrics alignment task.

`save_dir`: The directory to store the model checkpoints and hyperparameters. Need at least 10GB of disk space to store them!

### Inference & Evaluate

After training finished, you can execute our inference code to evaluate the model performance.
Or you can [Download our model]() to reproduce the result we mentioned in our paper.

```bash
# Alignment Evaluate
# set `--use-ctc-loss` flag if you used CTC Loss during traing phase.
# This script also computes the MAE metric!
python inference_alignment.py \
    -f [test_data] \
    --model-dir [model_dir] \
    --use-ctc-loss \
    --device [device_id] \

# Transcript Evaluate
# Get transcript result first. 
python inference_transcript.py \
    -f [test_data] \
    --model-dir [model_dir] \
    --output [output_file_path] \
    --device [device_id] \
    --use-groundtruth
# Evaluate the inference result file
python evaluate_transcript.py \
    -f [result_file_path]
```

`inference_transcript.py` writes the transcription to `[output_file_path]`. If the flag  `--use-groundtruth` is not set, the output file will not contain the groundtruth information, so you won't be able to evaluate it using `evaluate_transcript.py`.

If your samples in the json file do not have the `lyric` attribute (i.e., when you want to transcribe audios that do not have the groundtruth annotation), then you should not set this flag, as this will result in errors.

### Inference lyrics alignment without groundtruth

The above code for testing lyrics alignment models requires the groundtruth alignment annotation (so it can compute the MAE). If you don't have the groundtruth, run the following code:

```bash
python inference_alignment_nogt.py  
-f [test_data]  
--model-dir [model_dir]  
--use-ctc-loss  
--device [device_id]
```

The format of `test_data` is the same as in `inference_alignment.py`, but this time, the `on_offset` attribute is not required. On the other hand, `inference_alignment.py` automatically ignores samples that do not have the `on_offset` attribute (in order to handle `MIR1k_partial_align.json`).

## Some additional experiment results

Due to the page limit, we didn't put all the results to the paper. Here are some of them:

**MIR-1k** ALT result, w/ proposed method (the results we reported in the paper): **CER**=17.8%, **PER**=9.6%

**MIR-1k vocal** ALT result, w/ proposed method (only use vocal stems, can be viewed as using an oracle MSS model): **CER**=9.1%, **PER**=3.6%

More to come......

## **Contribution of each author on this repo**

Chon-In Leong (@navi0105) wrote about 80% of the training & evaluation code and the first version of dataset preprocessing code for the Opencpop dataset.

Yu-Chen Lin (@yuchen0515) wrote the code that detects overlapping songs between Opencpop's training set and MIR-1k.

Jun-You Wang (@york135) wrote about 20% of the training & evaluation code and all the remaining code on dataset preprocessing.