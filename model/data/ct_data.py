import pathlib
import random
import rawpy
import imageio
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to CT image slices.
    """

    def __init__(self, root, bounds=[-1000,1000]):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
        """
        self.root = str(root)
        self.examples = []
        self.bounds = bounds

        # get filenames
        files = list(pathlib.Path(str(root)+'/CT/').iterdir())
        for f in files:
            name = str(f)[len(str(root)+'/CT/'):].strip('.tif')
            self.examples += [name]
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname = self.examples[i]

        img = np.array(Image.open(self.root + '/CT/' + fname + '.tif')).astype(np.float32)
        mask = np.array(Image.open(self.root + '/mask/' + fname + '.tif')).astype(np.float32)

        img = np.clip(img, self.bounds[0], self.bounds[1])
        img = (img + self.bounds[1]) / (self.bounds[1] - self.bounds[0])
        mask = np.clip(mask, 0, 1)

        return (img, mask)
