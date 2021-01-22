from time import time
import numpy as np
import torch

class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim, n_layers, drop_prob):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class NetWrapper:
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim, n_layers, drop_prob):
        self.net = Net(input_dim, output_dim, hidden_dim, embedding_dim, n_layers, drop_prob)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)

    def learn(self, train_load, val_load, batch_size, learning_rate, n_epochs):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        print(f'\nTraining with {n_epochs} epochs:')

        self.net.train()
        clip = 5

        for epoch in range(n_epochs):
            # self.net.train()
            s_time = time()
            loss_total = 0
            # hidden = self.net.init_hidden(batch_size)
            for batch_idx, (inputs, labels) in enumerate(train_load):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # self.net.train()
                # hidden = tuple([e.data for e in hidden])
                # self.net.zero_grad()
                # print(inputs)
                # inputs = torch.rand(inputs.size())

                # output, hidden = self.net(inputs)
                output = self.net(inputs.float())
                # print(output)
                # print("output shape:", output.size())
                # print("labels shape:", labels.size())
                loss = criterion(output.squeeze(), labels.float())
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                # optimizer.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    self.net.zero_grad()
                    # val_h = model.init_hidden(batch_size)
                    val_losses = []
                    self.net.eval()
                    for inp, lab in val_load:
                        # print("inp:", inp.size())
                        # print("lab:", lab.size())
                        # val_h = tuple([each.data for each in val_h])
                        inp, lab = inp.to(self.device).float(), lab.to(self.device).float()
                        # print(inp)
                        # print(lab)
                        # out, val_h = self.net(inp, val_h)
                        out = self.net(inp)
                        # print(out)
                        val_loss = criterion(out.squeeze(), lab.float())
                        # print(val_loss)
                        val_losses.append(val_loss.item())

                    self.net.train()
                    print("Epoch: {}/{}...".format(epoch + 1, n_epochs),
                          "Step: {}...".format(batch_idx),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))
                #     if np.mean(val_losses) <= valid_loss_min:
                #         torch.save(model.state_dict(), './state_dict.pt')
                #         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                #                                                                                         np.mean(
                #                                                                                             val_losses)))
                #         valid_loss_min = np.mean(val_losses)
