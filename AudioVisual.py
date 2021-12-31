import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load("aefe.wav")
S = np.abs(librosa.stft(y))
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

fig, ax = plt.subplots(2, 1)

librosa.display.waveshow(y, sr=sr, ax=ax[0])
ax[0].set_title('Waveform')

img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
y_axis = 'log', x_axis = 'time', ax=ax[1])
ax[1].set_title('Spectrogram')

fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

plt.tight_layout()
plt.show()