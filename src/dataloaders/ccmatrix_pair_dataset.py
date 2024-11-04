from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import itertools


class CCMatrixPairDataset(Dataset):
    def __init__(self, language_pair='en-ru'):
        self.data = load_dataset("yhavinga/ccmatrix", language_pair, streaming=True)

    def __getitem__(self, idx):
        raise NotImplementedError("Streaming dataset does not support indexing")


def collate_fn(batch):
    sentence1_batch, sentence2_batch = zip(*batch)
    return list(sentence1_batch), list(sentence2_batch)
