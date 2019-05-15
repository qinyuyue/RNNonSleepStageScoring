from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, seqs, labels):
        assert len(seqs) == len(labels)
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]