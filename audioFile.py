import numpy as np
import soundfile as sf
import audioread as ar

data, sr = sf.read('aefe.wav')

rms = [np.sqrt(np.mean(block**2)) for block in
sf.blocks('aefe.wav', blocksize=30000, overlap=4000)]
print(rms)
sf.write('new_test.wav', data, sr)

import audioread as ar

with ar.audio_open('aefe.wav') as f:
    print(f.channels, f.samplerate, f.duration)
