from data_loaders.torchaudio_loader import TorchAudioDatasetWrapper
from models.lstm import *

torch.manual_seed(0)

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

    embedding_dim = 5000
    hidden_dim = 512
    n_layers = 2
    drop_prob = 0.5

    AudioData = TorchAudioDatasetWrapper(5000, 100, "raw_audio", (0.6, 0.2, 0.2))

    modelx = NetWrapper(input_dim, output_dim, hidden_dim, embedding_dim, n_layers, drop_prob)

    modelx.learn(AudioData.training_set, AudioData.validation_set, batch_size, learning_rate=0.005, n_epochs=5)