import pyworld as pw

def run_dio_pitch_detection(data, fs=48000):
    _f0, t = pw.dio(data, fs)
    f0 = pw.stonemask(data, _f0, t, fs)
    return f0
