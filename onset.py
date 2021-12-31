import librosa as lib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = lib.load("aefe.wav")
lib.onset.onset_detect(y=y,sr=sr,units='time')
o_env = lib.onset.onset_strength(y, sr=sr)
times = lib.times_like(o_env, sr=sr)
onset_frames = lib.onset.onset_detect(onset_envelope=o_env, sr=sr)

D = np.abs(lib.stft(y))
fig, ax = plt.subplots(nrows=2, sharex=True)

librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
x_axis='time', y_axis='log', ax=ax[0])
ax[0].set_title("Power Spectrogram")
ax[0].label_outer()

ax[1].plot(times, o_env, label="Onset strength")
ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r',
alpha=0.9,linestyle='--',label="Onsets")
ax[1].legend()
plt.show()
