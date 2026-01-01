import torch

def generate_xy_data(mapping, n, std):
    tmp_data_tensors = []
    tmp_label_tensors = []
    for k, v in mapping.items():
        tmp_data = torch.rand(n, 2)*std + torch.tensor(k)
        tmp_data_tensors.append(tmp_data)
        tmp_label_tensors.append(torch.tensor([v for p in range(n)]))
    result_data = torch.cat(tmp_data_tensors, 0)
    result_label = torch.cat(tmp_label_tensors, 0)
    return result_data, result_label


def generate_xy_data_improved(mapping, n, std):
    tmp_data_tensors = [torch.rand(n, 2) * std + torch.tensor(k) for k in mapping.keys()]
    tmp_label_tensors = [torch.full((n,), v) for v in mapping.values()]

    result_data = torch.cat(tmp_data_tensors, 0)
    result_label = torch.cat(tmp_label_tensors, 0)

    return result_data, result_label
