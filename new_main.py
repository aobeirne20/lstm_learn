from data_loaders.torchaudio_loader import TorchAudioDatasetWrapper
from models.new_lstm import *
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, show

torch.manual_seed(12)
torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    #Also the number of frames in the input NOT the sequential data input
    input_dim = 1
    #Also the number of frames in the output
    output_dim = 1

    #How many data points in one segment that is fed to the model in one load
    segment_length = 200
    #The model will only predict one datapoint ahead at a time (see output_dim). However, this many data points will be stored ahead for auto-regression.
    prediction_length = 25

    batch_size = 20


    hidden_dim =  50
    n_layers = 2


    AudioData = TorchAudioDatasetWrapper(segment_length, prediction_length, "raw_audio", (0.6, 0.1, 0.3))
    training_load = torch.utils.data.DataLoader(AudioData.training_set, batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
    validation_load = torch.utils.data.DataLoader(AudioData.validation_set, batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
    testing_load = torch.utils.data.DataLoader(AudioData.testing_set, 1, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

    data, data_shift1 = next(iter(testing_load))
    data_reshaped = torch.squeeze(torch.reshape(data, (1, 1, segment_length)))
    data_reshaped = data_reshaped.cpu().numpy()
    x = np.linspace(0, data_reshaped.size - 1, num=data_reshaped.size)
    print(x)

    plt.plot(x, data_reshaped)


    model = LSTMWrapper(input_dim, output_dim, hidden_dim, n_layers, batch_size)
    model.learn(training_load, validation_load, learning_rate=0.00001, n_epochs=20, print_every=1)

    prediction = model.test(data)
    plot(x, data[0, :, 0]); plot(x, torch.squeeze(prediction[0, :, 0]).cpu().detach().numpy()); plt.show()
