import torch, torchaudio
import os

#We really should consider an iterable style dataset, but thats too much for now. The iterable dataset will allow the
#loading of data that is not loaded in its entiriety from the beginning, allowing data to be "streamed" into the
#dataset as needed. Also look up torch.utils.data.TensorDataset (actually maybe not)./

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Torchaudio shit
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")  # The new interface


class TorchAudioDatasetWrapper:
    def __init__(self, segment_length, label_length, audio_file_location, tvt_split_ratio):
        #Feature dim/input dim is ALWAYS 1. The segment length controls the 1 feature inputs in chronological order.
        soundfiles_dir = os.path.join(os.getcwd(), audio_file_location)
        print(f"Searching {soundfiles_dir} for appropriate sound files \n")
        audio_file_names = os.listdir(soundfiles_dir)
        loaded_sequences = []
        loaded_segments = []
        for file_name in audio_file_names:
            file_path = os.path.join(os.getcwd(), audio_file_location, file_name)
            audio_info = torchaudio.backend._soundfile_backend.info(file_path)
            print(f"Loaded audio file {file_name}: {audio_info.num_channels} channel(s) @{audio_info.sample_rate}Hz, {audio_info.num_frames} frames long")
            audio_sequence = torchaudio.backend._soundfile_backend.load(file_path, normalize=True)[0]
            if torch.cuda.is_available():
                audio_sequence = audio_sequence.to('cuda:0')
                print(f"Transferred {file_name} to CUDA successfully")
            if audio_info.num_channels > 1:
                split_sequences = torch.split(audio_sequence, 1)
                for sequence in split_sequences:
                    loaded_sequences.append(torch.squeeze(sequence))
            else:
                loaded_sequences.append(torch.squeeze(audio_sequence))

        print()
        num_eliminated = 0
        for loaded_sequence in loaded_sequences:
            print(f"Converting sequence of {list(loaded_sequence.size())[0]} length into {int(list(loaded_sequence.size())[0]/segment_length)} of {segment_length} long")
            split_segments = torch.split(loaded_sequence, segment_length)

            for n in range(0, len(split_segments)-1):
                current_segment = split_segments[n]
                next_segment = split_segments[n+1]
                if (list(current_segment.size())[0] == segment_length) & (list(next_segment.size())[0] > label_length):
                    loaded_segments.append(SegmentPair(current_segment, next_segment[0:label_length]))
                    if (n == len(split_segments)-2) & (list(next_segment.size())[0] == segment_length):
                        num_eliminated += 1
                else:
                    num_eliminated += 1
        print(f"\nCreated {len(loaded_segments)} data/label pairs, eliminated {num_eliminated} due to insufficient data")

        num_training_sets = int(tvt_split_ratio[0]*len(loaded_segments))
        num_validation_sets = int(tvt_split_ratio[1]*len(loaded_segments))
        num_testing_sets = int(len(loaded_segments) - num_training_sets - num_validation_sets)

        self.training_set = TorchAudioDataset(loaded_segments[:num_training_sets - 1])
        self.validation_set = TorchAudioDataset(loaded_segments[num_training_sets: num_training_sets + num_validation_sets - 1])
        self.testing_set = TorchAudioDataset(loaded_segments[num_training_sets + num_validation_sets:])

        print(f"\nSplit loaded data:")
        print(f"Training data: {num_training_sets} pairs, indices [0:{num_training_sets - 1}]")
        print(f"Validation data: {num_validation_sets} pairs, indices [{num_training_sets}:{num_training_sets + num_validation_sets - 1}]")
        print(f"Testing data: {num_testing_sets} pairs, indices [{num_training_sets + num_validation_sets}:{num_training_sets + num_validation_sets + num_testing_sets - 1}]")
        print(f"--------------------------------------------------------------------------------")
        print(f'Total data: {len(loaded_segments)} pairs, indices [0:{len(loaded_segments)-1}]')


class TorchAudioDataset(torch.utils.data.Dataset):
    def __init__(self, segment_set):
        self.segment_set = segment_set

    def __len__(self):
        return len(self.segment_set)

    def __getitem__(self, idx):
        segment = self.segment_set[idx].segment_data
        prediction = self.segment_set[idx].prediction_label
        sample = {'segment': segment, 'prediction': prediction}
        return sample


class SegmentPair:
    __slots__ = ['segment_data', 'prediction_label']

    def __init__(self, segment_data, prediction_label):
        self.segment_data = segment_data
        self.prediction_label = prediction_label



