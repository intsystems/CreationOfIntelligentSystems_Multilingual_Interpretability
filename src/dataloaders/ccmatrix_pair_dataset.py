from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import itertools


class CCMatrixPairDataset(IterableDataset):
    def __init__(self, language_pair='en-ru'):
        self.language_pair = language_pair
        self.data = load_dataset("yhavinga/ccmatrix", self.language_pair, streaming=True)

    def __iter__(self):
        for item in self.data['train']:
            sentence1, sentence2 = item['translation'].values()
            yield sentence1, sentence2

def collate_fn(batch):
    sentence1_batch, sentence2_batch = zip(*batch)
    return list(sentence1_batch), list(sentence2_batch)
