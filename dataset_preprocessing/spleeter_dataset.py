import librosa
import numpy as np
import soundfile as sf
import torch
import sys, os
from tqdm import tqdm

from spleeter.separator import Separator
import warnings
separator = Separator('spleeter:2stems')
warnings.filterwarnings('ignore')

def infer_vocal_spleeter(mix_np):
    waveform = np.expand_dims(mix_np, axis=1)
    prediction = separator.separate(waveform)
    ret_voc = librosa.core.to_mono(prediction["vocals"].T)
    ret_voc = np.clip(ret_voc, -1.0, 1.0)
    return ret_voc

if __name__ == "__main__":
    audio_dir = sys.argv[1]
    separated_dir = sys.argv[2]

    if not os.path.exists(separated_dir):
        os.mkdir(separated_dir)

    
    for audio_name in tqdm(os.listdir(audio_dir)):
        audio_path = os.path.join(audio_dir, audio_name)

        y, _ = librosa.load(audio_path, sr=44100, mono=True)

        output = infer_vocal_spleeter(y)

        output_path = os.path.join(separated_dir, audio_name)
        sf.write(
            output_path,
            output,
            44100,
            "PCM_16",
        )