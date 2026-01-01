import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors

def visualize_samples(data, label):
    datas = []
    for level in np.unique(label):
        datas.append(data[label == level])
    plt.figure(figsize=(4,4))
    for level in np.unique(label):
        plt.scatter(datas[level][:, 0], datas[level][:, 1], edgecolor="#333", label="Class " + str(level))
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()
    plt.show()


@torch.no_grad()
def visualize_classification(data, label, colors, predicted_class):
    plt.figure(figsize=(6, 6))
    for i in range(len(colors)):
        class_data = data[label == i]
        plt.scatter(class_data[:, 0], class_data[:, 1], edgecolor="#333", label=f"Class {i}", color=colors[i])
    labelize()
    # Map predicted classes to colors
    output_image = np.zeros((predicted_class.shape[0], predicted_class.shape[1], 4))  # RGBA image
    for i in range(len(colors)):
        output_image[torch.transpose(predicted_class, 0, 1) == i] = matplotlib.colors.to_rgba(colors[i])
    plt.imshow(output_image, origin='lower', extent=(0.0, 3.5, 0.0, 3.5))
    plt.grid(False)


def labelize():
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()