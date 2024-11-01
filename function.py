from torch.utils.data import DataLoader, Dataset as TorchDataset

class CustomDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, dict):
            return item
        return {k: item[k] for k in self.dataset.column_names}