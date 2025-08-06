import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def save_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=22050)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_DB, sr=sr, hop_length=512, cmap='magma')
    plt.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Test run
save_spectrogram("data/deep/01 - Breather.mp3", "spectrograms/deep1.png")
