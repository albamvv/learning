import numpy as np
imoprt matplotlib.pyplot as plt
import librosa

audio_path ='name.mp3'

# sr -sample rate
y, sr = librosa.load(audio_path, sr=None)
print(type(y))
print()"y->",y.shape)
print('sr-> ',sr) # Hz  48000 Hz --> 48 kHz --> 1/sec

plt.figure(figsize(14,5),dpi=150)
plt.plot(y)
pl.xlabel('Time - Samples')
plt.ylabel('Amplitude')
