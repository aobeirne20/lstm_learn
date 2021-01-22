import sys
sys.path.append("..")
from util import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("specify input .wav file!"); sys.exit()
    if len(sys.argv) < 3:
        print("specify output .npy file!"); sys.exit()
    save_signal(load_wave(sys.argv[1]), sys.argv[2])
