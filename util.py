import sys
import numpy as np
import soundfile as sf
import sounddevice as sd

def load_wave(filename):
    data, fs = sf.read(filename, dtype='float32')
    print("fs: " + str(fs))
    return data[:, 0] if len(data.shape) > 1 else data

def save_signal(signal, filename):
    np.save(filename, signal)

def load_signal(filename):
    return np.load(filename).astype(float)

def play(data, fs=48000):
    data = np.array(data).astype(np.float32)
    data -= np.mean(data)
    data /= np.max(np.abs(data))
    sd.play(data, fs)
    sd.wait()

def record(seconds, fs=48000):
    print("recording *")
    data = sd.rec(int(seconds * fs), samplerate=fs, channels=2)[:,0]
    sd.wait()
    print("done *")
    return data
