import librosa as lib
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
y, sr = lib.load("aefe.wav")
S = np.abs(lib.stft(y))
freqs = librosa.fft_frequencies(sr)
harms = [1,2,3,4] # harmonics, 1st is S itself
w = [1.0, 0.5, 0.33, 0.25] # must be same len as harmonics
S_sal = lib.salience(S, freqs, harms, w, fill_value=0)
print(S_sal.shape)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=False)
librosa.display.specshow(lib.amplitude_to_db(S, ref=np.max),
sr=sr, y_axis='log',x_axis='time', ax=ax[0])
ax[0].set_title("Magnitude Spectrogram")
ax[0].label_outer()

img = librosa.display.specshow(lib.amplitude_to_db(S_sal, ref=np.max),
sr=sr, y_axis='log',x_axis='time', ax=ax[1])
ax[1].set_title("Salience Spectrogram")
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()