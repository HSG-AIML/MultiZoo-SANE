import torch

from torch.utils.data import Dataset


#####################################################################
# Define Dataset class
#####################################################################
class DatasetRandom(Dataset):
    """
    dataset returns random data of the same shape as
    dataset_token, w/ or w/o transformations
    extremely cheap, to identify bottlnecks.
    """

    def __init__(self, len, sequencelen, tokensize):
        super(DatasetRandom, self).__init__()

        self._len = len
        self.sequencelen = sequencelen
        self.tokensize = tokensize

    def __len__(self):
        return self._len

    def __getitem__(self, index):

        w = torch.randn(self.sequencelen, self.tokensize)
        m = torch.randn(self.sequencelen, self.tokensize)
        p = torch.randint(low=0, high=250, size=(self.sequencelen, 3))

        w = w.to(torch.float16)
        m = m.to(torch.bool)
        p = p.to(torch.int16)
        return w, m, p
