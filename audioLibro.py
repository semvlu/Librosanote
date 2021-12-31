import librosa as lib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = lib.load("aefe.wav")
S = np.abs(lib.stft(y))

stream = lib.stream("aefe.wav", block_length=256,
frame_length=4096, hop_length=1024)

for y_block in stream:
    m_block = lib.feature.melspectrogram(y_block, sr=sr,
    n_fft=2048, hop_length=2048, center=False)

fig, ax = plt.subplots()
S_dB = lib.power_to_db(m_block, ref=np.max)

img = lib.display.specshow(S_dB, x_axis='time',
y_axis='mel', sr=sr, fmax=10000, ax=ax)

fig.colorbar(img,ax=ax,format="%+2.0f dB")
ax.set_title('Mel-Freq Spectrogram')
plt.show()

