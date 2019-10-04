import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, nibabel
import sys, getopt
import PIL
from PIL import Image
import imageio
import scipy.misc
from torch.utils import data
import torch
import random
from .augmentations import augment_spatial, Compose, RandomRotate, PaddingCenterCrop
from skimage import transform

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = -data_sample
    return data_sample

class CTData(data.Dataset):

    def __init__(self,
                 root='gs://vector-data-bucket-smh/C_Spine_Hackathon',
                 split='train',
                 augmentations=None,
                 target_size=(512, 512)
                 t="3D"):
        self.target_size = target_size
        self.ROOT_PATH = root
        self.augmentations = augmentations
        self.split = split
        self.list = self.read_files()

    def read_files(self):
        d = []
        if t=="3D":
            for i in os.listdir(os.path.join(root, "Positive_cases")):
                d.append(i)
            for i in os.listdir(os.path.join(root, "Negative_cases")):
                d.append(i)
        return d

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i): # i is index
        
        if t=="3D":
            if self.list[i].startswith("P"):
                path = os.listdir(os.path.join(root, "Positive_cases", self.list[i]))
            else:
                path = os.listdir(os.path.join(root, "Negative_cases", self.list[i]))

            img_vol = torch.tensor(1, len(path) + (16-len(path)%16), target_size[0], target_size[1])
            seg_vol = torch.tensor(1, len(path) + (16-len(path)%16), target_size[0], target_size[1])
            
            if self.list[i].startswith("P"):
                enum = os.path.join(path, "full_CT_images")
            else:
                enum = path

            for x, f in enumerate(enum):
                full_img_path = os.path.join(path, "full_CT_images", f) if self.list[i].startswith("P") else os.path.join(path, f)
                full_seg_path = os.path.join(path, "masks", f) if self.list[i].startswith("P") else None
                img = np.array(Image.open(full_img_path))
        
                if full_seg_path != None and os.path.exists(full_seg_path):
                    seg = np.array(Image.open(full_seg_path))
                else:
                    seg = np.zeros((512,512))

                if img.min() > 0:
                    img -= img.min()
        
                img, seg = self.augmentations(img.astype(np.uint32), seg,astype(np.uint8))
                img = augment_gamma(img)

                if self.img_norm:
                    mu = img.mean()
                    sigma = img.std()
                    img = (img - mu) / (sigma+1e-10)


                img, seg = self._transform(img, seg)
                img_vol[0][x] = img
                seg_vol[0][x] = seg

            for y in range(16-len(path)%16):
                img_vol[0][len(path) + y] = torch.zeros(512,512)
                seg_vol[0][len(path) + y] = torch.zeros(512,512)

            return img_vol, seg_vol

    def _transform(self, img, mask):
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        return img, mask


    def random_elastic_deformation(self, image, alpha, sigma, mode='nearest',
                                   random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
    ..  [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        assert len(image.shape) == 3

        if random_state is None:
            random_state = np.random.RandomState(None)

        height, width, channels = image.shape

        dx = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        indices = (np.repeat(np.ravel(x+dx), channels),
                np.repeat(np.ravel(y+dy), channels),
                np.tile(np.arange(channels), height*width))

        values = map_coordinates(image, indices, order=1, mode=mode)

        return values.reshape((height, width, channels))

if __name__ == '__main__':

    DATA_DIR = 'gs://vector-data-bucket-smh/C_Spine_Hackathon'
    augs = Compose([])
    dataset = CTData(DATA_DIR, augmentations=augs)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for idx, batch in enumerate(dloader):
        img, mask = batch['image'], batch['mask']
        print(mask.shape, img.shape, mask.max(), mask.min(), img.max(), img.min())
