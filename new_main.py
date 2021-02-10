from data_loaders.torchaudio_loader import TorchAudioDatasetWrapper
from models.new_lstm import *

torch.manual_seed(0)
torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    #Also the number of frames in the input NOT the sequential data input
    input_dim = 1
    #Also the number of frames in the output
    output_dim = 1

    #How many data points in one segment that is fed to the model in one load
    segment_length = 1000
    #The model will only predict one datapoint ahead at a time (see output_dim). However, this many data points will be stored ahead for auto-regression.
    prediction_length = 20

    batch_size = 20


    hidden_dim =  50
    n_layers = 2


    AudioData = TorchAudioDatasetWrapper(150, 100, "raw_audio", (0.9, 0.05, 0.05))
    training_load = torch.utils.data.DataLoader(AudioData.training_set, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    validation_load = torch.utils.data.DataLoader(AudioData.validation_set, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    testing_load = torch.utils.data.DataLoader(AudioData.testing_set, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

    model = LSTMWrapper(input_dim, output_dim, hidden_dim, n_layers, batch_size)
    model.learn(training_load, validation_load, learning_rate=0.0005, n_epochs=1000, print_every=1)