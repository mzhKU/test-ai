import torch.utils.data as data

class CustomDataset(data.Dataset):
    def __init__(self, data_points, labels, mapping, n, std):
        super().__init__()
        self.mapping = mapping
        self.n =n
        self.data = data_points
        self.labels = labels
    
    def __len__(self):
        return len(self.mapping)*self.n

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.labels[idx]
        return data_point, data_label