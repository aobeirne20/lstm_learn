import os
import numpy as np
import random
import torch


class AudioSnipDataset(torch.utils.data.Dataset):
    def __init__(self, directory, seg_length, ans_length, type):
        data_sets = []
        temp_segments = []
        temp_answers = []
        self.type = type

        for file in os.listdir(directory):
            if file.endswith(".signal"):
                print("loading", file, self.type)
                data_sets.append(np.fromfile(os.path.join(directory, file), dtype=float))

        self.row_num = 0
        for data_array in data_sets:
            for index in range(0, data_array.size - 1 - seg_length - ans_length):
                self.row_num += 1
                temp_segments.append(data_array[index:index + seg_length])
                temp_answers.append(data_array[index + seg_length:index + seg_length + ans_length])

        x_data_np = np.vstack(temp_segments)
        y_data_np = np.vstack(temp_answers)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.x_data = torch.tensor(x_data_np, dtype=torch.float32).to(self.device)
        self.y_data = torch.tensor(y_data_np, dtype=torch.float32).to(self.device)

        self.n_train_samples = int(0.6 * self.row_num)
        self.n_val_samples = int(0.2 * self.row_num)
        self.n_test_samples = self.row_num - self.n_train_samples - self.n_val_samples
        
        splits = [self.n_train_samples, self.n_val_samples, self.n_test_samples]
        self.x_data_train, self.x_data_val, self.x_data_test = torch.split(self.x_data, splits)
        self.y_data_train, self.y_data_val, self.y_data_test = torch.split(self.y_data, splits)

    def __len__(self):
        if self.type == "train":
            return self.n_train_samples
        elif self.type == "val":
            return self.n_val_samples
        elif self.type == "test":
            return self.n_test_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == "train":
            segment = self.x_data_train[idx, :]
            answer = self.y_data_train[idx]
            return segment, answer
        elif self.type == "val":
            segment = self.x_data_val[idx, :]
            answer = self.y_data_val[idx]
            return segment, answer
        elif self.type == "test":
            segment = self.x_data_test[idx, :]
            answer = self.y_data_test[idx]
            return segment, answer
