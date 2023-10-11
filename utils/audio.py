import librosa

def load_audio_file(file, audio_type: int=0):
    # audio_type: 0 => mono; 1 => mixture; 2 => mixture, but vocal only
    is_mono = audio_type == 0
    
    batch = {}
    speech, _ = librosa.load(file, sr=16000, mono=is_mono)
    
    if audio_type == 0:
        batch["speech"] = speech
    elif audio_type == 1:
        batch["speech"] = (speech[0] + speech[1]) / 2
    elif audio_type == 2:
        batch["speech"] = speech[1]
    else:
        raise ValueError("audio_type must be 0, 1, or 2")
    
    batch["sampling_rate"] = 16000
    return batch
