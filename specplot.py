import librosa as lib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = lib.load("sample.wav")
y_harm, y_perc = lib.effects.hpss(y)
S_harm = np.abs(lib.stft(y_harm))
S_perc = np.abs(lib.stft(y_perc))

fig, ax = plt.subplots(2, 1)
img = librosa.display.specshow(librosa.amplitude_to_db(S_harm, ref=np.max),
y_axis='log', x_axis='time',ax=ax[0])
ax[0].set_title("Harmonics Spectrogram")
fig.colorbar(img, ax=ax[0], format="%+2.0f dB")

img1 = librosa.display.specshow(librosa.amplitude_to_db(S_perc, ref=np.max),
y_axis='log', x_axis='time',ax=ax[1])
ax[1].set_title("Percussion Spectrogram")
fig.colorbar(img1, ax=ax[1], format="%+2.0f dB")
plt.show()