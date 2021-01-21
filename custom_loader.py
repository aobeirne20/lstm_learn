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
                data_sets.append(np.fromfile(os.path.join(directory, file), dtype=float))

        self.row_num = 0
        for data_array in data_sets:
            for index in range(0, data_array.size - 1 - seg_length - ans_length):
                self.row_num += 1
                temp_segments.append(data_array[index:index + seg_length])
                temp_answers.append(data_array[index + seg_length + 1:index + seg_length + ans_length + 1])

        x_data_np = np.vstack(temp_segments)
        y_data_np = np.vstack(temp_answers)

        self.x_data = torch.tensor(x_data_np, dtype=torch.float32).to("cuda:0")
        self.y_data = torch.tensor(y_data_np, dtype=torch.float32).to("cuda:0")

        self.x_data_train, self.x_data_val, self.x_data_test = torch.split(self.x_data, [int(0.6 * self.row_num)+1, int(0.2 * self.row_num), int(0.2 * self.row_num)])
        self.y_data_train, self.y_data_val, self.y_data_test = torch.split(self.y_data, [int(0.6 * self.row_num)+1, int(0.2 * self.row_num),
                                                                                         int(0.2 * self.row_num)])

    def __len__(self):

        if self.type == "train":
            return int(0.6 * self.row_num)+1
        elif self.type == "val":
            return int(0.2 * self.row_num) + 1
        elif self.type == "test":
            return int(0.2 * self.row_num) + 1

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





