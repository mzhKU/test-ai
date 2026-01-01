import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.activation = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class ClassifierSingleHiddenLayerTanh(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

class ClassifierTwoHiddenLayerTanh(nn.Module):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs):
        super().__init__()
        self.act_fn = nn.Tanh()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3 = nn.Linear(num_hidden2, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        return x
    
class ClassifierTwoHiddenLayerReLU(nn.Module):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs):
        super().__init__()
        self.act_fn = nn.ReLU()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3 = nn.Linear(num_hidden2, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        return x

class ClassifierFiveHiddenLayerTanh(nn.Module):
    def __init__(self, num_inputs, h1, h2, h3, h4, h5, n_outputs):
        super().__init__()
        self.act_fn = nn.Tanh()
        self.linear1 = nn.Linear(num_inputs, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, h3)
        self.linear4 = nn.Linear(h3, h4)
        self.linear5 = nn.Linear(h4, h5)
        self.linear6 = nn.Linear(h5, n_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        x = self.act_fn(x)
        x = self.linear4(x)
        x = self.act_fn(x)
        x = self.linear5(x)
        x = self.act_fn(x)
        x = self.linear6(x)
        return x   