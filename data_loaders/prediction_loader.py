import os
import numpy as np
import random
import torch

from util import *

class AudioData:
    DATA_DIR = os.path.join(os.getcwd(), "new_data")

    def __init__(self, input_dim, output_dim, batch_size):
        train_set = AudioSnipDataset(os.path.join(AudioData.DATA_DIR, "train"), input_dim, output_dim)
        self.train_load = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

class AudioSnipDataset(torch.utils.data.Dataset):
    def __init__(self, directory, seg_length, ans_length):
        self.seg_length, self.ans_length = seg_length, ans_length
        self.data_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".npy")]
        self.n_data_files = len(self.data_files)
        self.data_file_sizes = [len(load_signal(file)) for file in self.data_files]
        print(f"found {self.n_data_files} files in data directory: {self.data_files}")

        n_frames = sum([len(load_signal(file)) for file in self.data_files])
        self.n_examples = n_frames - self.n_data_files * (seg_length + ans_length)
        print("n frames:", n_frames)
        print("n examples:", self.n_examples)

        self.cached_file = None
        self.cached_data = None
        self.total_requests = 0

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        # self.total_requests += 1
        # print("total req:", self.total_requests)
        # print("getting item #", idx)
        file = self.get_file_for_idx(idx)
        data = self.cached_data if file == self.cached_file else load_signal(file)
        self.cached_file = file
        self.cached_data = data
        return data[idx:idx + self.seg_length], data[idx + self.seg_length:idx + self.seg_length + self.ans_length]

    def get_file_for_idx(self, idx):
        file_idx = 0
        frames_passed = 0
        while idx + self.seg_length + self.ans_length + frames_passed > self.data_file_sizes[file_idx]:
            frames_passed += self.data_file_sizes[file_idx] - (self.seg_length + self.ans_length)
            file_idx += 1
        return self.data_files[file_idx]
