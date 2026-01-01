import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

from data_generator import generate_xy_data, generate_xy_data_improved
from classifier import ClassifierSingleHiddenLayerTanh, \
        ClassifierTwoHiddenLayerTanh, \
        ClassifierFiveHiddenLayerTanh, \
        Perceptron, \
        ClassifierTwoHiddenLayerReLU
from visualize import visualize_classification, visualize_samples
from utilities import train_model, train_with_metrics, calculate_mesh_grid
from mappings import xor_mapping, xy_mapping, layer_mapping, layer_mapping_two_levels
from dataset import CustomDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

doTrain     = True
colors      = ['C0', 'C1', 'C2', 'C3']
mapping     = xy_mapping
n_train     = 10000
n_test      = 20
std         = 0.9
batch_size  = 256

loss_module = nn.CrossEntropyLoss() # Applies softmax to convert raw logits to probabilities

# model = ClassifierSingleHiddenLayerTanh(num_inputs = 2, num_hidden = 1, num_outputs=4)
model = ClassifierTwoHiddenLayerTanh(num_inputs=2, num_hidden1=12, num_hidden2=8, num_outputs=4)
# model = ClassifierTwoHiddenLayerReLU(num_inputs=2, num_hidden1=12, num_hidden2=8, num_outputs=4)
# model = ClassifierFiveHiddenLayerTanh(2, 10, 10, 10, 10, 10, 4)
# model = Perceptron(num_inputs=2, num_outputs=4)

# optimizer   = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer   = torch.optim.Adam(model.parameters(), lr=0.01)
model.to(device)

train_data, train_labels = generate_xy_data_improved(mapping, n_train, std)
test_data, test_labels = generate_xy_data_improved(mapping, n_test, std)

# Build a validation split from the generated training data
val_fraction = 0.1
num_samples = train_data.size(0)
perm = torch.randperm(num_samples)
split = int(num_samples * (1.0 - val_fraction))
train_idx = perm[:split]
val_idx = perm[split:]

train_X = train_data[train_idx]
train_y = train_labels[train_idx]
val_X = train_data[val_idx]
val_y = train_labels[val_idx]

train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
test_data_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

if doTrain:
    train_with_metrics(device, model, optimizer, train_data_loader, val_data_loader, loss_module, num_epochs=40, early_stop_threshold=0.02)
    state_dict = model.state_dict()
    torch.save(state_dict, "mymodel.tar")
else:
    state_dict = torch.load("mymodel.tar")
    model.load_state_dict(state_dict)

# Convert test tensors to numpy arrays for visualization
test_data = test_data.cpu().numpy() if isinstance(test_data, torch.Tensor) else test_data
test_labels = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels

# visualize_samples(test_data, test_labels)

model.to(device)
xx1, xx2 = calculate_mesh_grid(-0.5, 3.5, step=0.01, device=device)
arguments = torch.stack([xx1, xx2], dim=-1)

predicted_class = model(arguments)
predicted_class = torch.argmax(predicted_class, dim=-1)  # Get the predicted class

print(predicted_class.size())
visualize_classification(test_data, test_labels, colors, predicted_class)
plt.show()