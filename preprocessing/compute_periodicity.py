import sys, os, shutil
import matplotlib.pyplot as plt
import pyworld as pw

sys.path.append("..")
from util import *

def run_dio_pitch_detection(data, fs=96000):
    f0, t = pw.dio(data, fs, frame_period=1000 / fs, f0_floor=60, f0_ceil=1200)
    f0 = pw.stonemask(data, f0, t, fs)
    return f0

def create_target_dataset(input_dir, target_dir):
    input_files = [file for file in os.listdir(input_dir) if file.endswith(".npy")]
    shutil.rmtree(target_dir); os.mkdir(target_dir)
    for file in input_files:
        print("processing ", file)
        input_data = load_signal(os.path.join(input_dir, file))
        target_data = run_dio_pitch_detection(input_data)[:-1]
        assert len(input_data) == len(target_data)
        plt.plot(input_data); plt.plot(target_data); plt.show()
        save_signal(target_data, os.path.join(target_dir, file))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("specify input directory!")
        sys.exit(1)
    if len(sys.argv) < 3:
        print("specify target directory!")
        sys.exit(1)

    create_target_dataset(sys.argv[1], sys.argv[2])
