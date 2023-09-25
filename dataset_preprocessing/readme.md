# Dataset preprocessing

This folder contains 1) the character-level alignment of a subset of the MIR-1k dataset and 2) source code for dataset preprocessing.

For convenient, we refer to such a character-level alignment dataset as *MIR-1k-partial-align*, as only 17 out of 86 songs were annotated with character-level alignments.

**Contribution of each author.** Chon-In Leong (@navi0105) wrote the first version of dataset preprocessing code for the Opencpop dataset. Yu-Chen Lin (@yuchen0515) wrote the code that detects overlapping songs between Opencpop's training set and MIR-1k. The remaining works on dataset preprocessing were done by Jun-You Wang (@york135).

## Time-aligned MIR-1k dataset (MIR-1k-partial-align)

**Acknowledgement.** We would like to thank Meng-Hua Yu for annotating the chracter-level alignment of 17 songs in the MIR-1k dataset. This for the first time allows us to evaluate Mandarin lyrics alignment performance on polyphonic data (mixture of vocals & accompaniments).

The file "MIR1k_partial_align.json" contains the annotation of *MIR-1k-partial-align*. For the audio file, please download the MIR-1k dataset by yourself (from [MIR Corpora](http://mirlab.org/dataset/public/)). We use the files in the "UndividedWavfile/" directory of the dataset.

#### Data structure

The JSON file contains a list. Each element in the list represents one song (one sample), and has the data structure of dictionary. The keys are "**song_id**", "**lyric**", and optionally "**on_offset**". If one song has the character-level alignment label, then it would has the "on_offset" attribute. The detailed descriptions are as follows:

- "**song_id**": A string. Indicate the basename of the song (e.g. "khair_1.wav").

- "**lyric**": A string. Indicate the lyrics of the song, which is obtained from MIR-1k's original labels (in "Lyrics/"). All the lyrics are converted to Simplified Chinese.

- "**on_offset**": A list of [onset, offset]. Indicate the character-level alignment labels. The length of "**on_offset**" is always the same as the length of "**lyric**".

Note that there are actually 110 songs in the original "UndividedWavfile/", but we only include 86 songs here. We filter out songs that either 1) has non-Mandarin lyrics or 2) overlaps with Opencpop's training set.

#### Regarding the accuracy of the labels

We have manually verified the lyrics of the 17-song subset during the annotation process. There are indeed some errors in the original MIR-1k's lyrics labels (and we have already corrected them). However, we do not have the time to verify the lyrics of the remaining 69 songs. Therefore, there may be some potential errors in these songs' lyrics labels.

## Dataset preprocessing

First, we have to manually augment the Opencpop dataset with the Musdb-18 dataset. Please first obtain the audios of the two datasets by yourself (we are not supposed to re-distribute them).

### Data augmentation

Run this command to augment instrument tracks to augment Opencpop:

```
python mix_with_musdb.py [audio_dir] [augment_dir] [musdb_dir] [snr]
```

`audio_dir`: The directory to the audio files to be augmented. It will use ``os.listdir(audio_dir)`` to get the list of audio files to be augmented (i.e., will not traverse the directory recursively).

``augment_dir``: The directory that the augmented audios will be written to. In this process, the file basename will not be modified.

``musdb_dir``: The directory to Musdb-18's test set (NOT the whole dataset!). It will find those ``accompaniment.wav`` files and randomly choose one for augmentation for each audio.

Actually, in our experiment, we left 10 songs (all songs after "The Easton Ellises (Baumi) - SDRNR" in alphabetical order) in Musdb-18's test set out for some preliminary tests. You can decide if you also want to adopt this setting. If so, just remove these 10 songs from the `musdb_dir`.

``snr``: The desired SNR value.

Our experiments take three augmented versions (w/ SNR of 0, -5, -10) for training. To reproduce it, you have to run the above code for 3 times, each time with different SNR.

### Run HT Demucs (preprocessing)

```
python demucs_dataset.py [audio_dir] [separated_dir]
```

`audio_dir`: The directory to the input audio files.

`separated_dir`: The directory that the extracted vocals will be written to.

This should be applied to both Opencpop and MIR-1k.

### Run Spleeter (preprocessing, for ablation studies)

```
python spleeter_dataset.py [audio_dir] [separated_dir]
```

The arguments are the same as ``demucs_dataset.py``. To run this code, you have to install the Spleeter package (see [https://github.com/deezer/spleeter](https://github.com/deezer/spleeter)), which is not included in ``requirements.txt`` (as this was only used in ablation studies). Again, this should be applied to both Opencpop and MIR-1k.

### Add absolute song_path attributes

In the final step, we have to add the absolute "**song_path**" attribute to every sample in the json files, so our training scripts can understand where the audios are actually stored in solely from those json files.

```
python replace_path.py [data_path] [output_path] [target_dir]
```

`data_path`: The path to the input json file.

`output_path`: The path that the modified json file (after adding song_path for each sample) will be written to.

`target_dir`: The directory to the audio files. The song_path will be ``str(Path(target_dir).joinpath(song_id))``, where ``song_id`` is the song_id attribute of each data sample. Make sure that this leads to the correct absolute path of the corresponding audio.

As a reminder, to reproduce our experiments (exclude ablation studies), you need 3 JSON files (augmented with SNR=0, -5, -10, and use HT Demucs to extract vocals) for the training and validation set of Opencpop, 1 JSON file (without data augmentation) for Opencpop's test set, 1 JSON file (without data augmentation, and use HT Demucs to extract vocals) for the MIR-1k dataset.

**That's it :)**


