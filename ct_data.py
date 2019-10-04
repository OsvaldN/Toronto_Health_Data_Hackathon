import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys, getopt
import PIL
import numpy as np
from PIL import Image
import imageio
import scipy.misc
from torch.utils import data
import torch
import random
from skimage import transform

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from google.colab import auth
import tensorflow as tf

tf.enable_eager_execution()
auth.authenticate_user()

class CTData(data.Dataset):

    def __init__(self,
                 root='gs://vector-data-bucket-smh/C_Spine_Hackathon',
                 split='train',
                 augmentations=None,
                 target_size=(512, 512),
                 t="3D"):
        self.target_size = target_size
        self.ROOT_PATH = root
        self.augmentations = augmentations
        self.split = split
        self.list = self.read_files()
        self.list = [x for x in self.list if not x.startswith('.')]
        self.t = t

    def read_files(self):
        root = self.ROOT_PATH #TODO clean up
        d = []
        if True: #TODO remove indentation
            for i in tf.io.gfile.listdir(root + "/Positive_cases"):
                d.append(i)
            for i in tf.io.gfile.listdir(root + "/Negative_cases"):
                d.append(i)
        return d

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i): # i is index
        root = self.ROOT_PATH #TODO clean up 
        if True: #TODO remove indentation
            if self.list[i].startswith("P"):
                #path = tf.io.gfile.listdir(root + "/Positive_cases/" + self.list[i])
                path = root + "/Positive_cases/" + str(self.list[i])
            else:
                #path = tf.io.gfile.listdir(root + "/Negative_cases/" + self.list[i])
                path = root + "/Negative_cases/" + str(self.list[i])

            if self.list[i].startswith("P"):
                print(path)
                #enum = os.path.join(path, "full_CT_images")
                enum = path + "full_CT_images/"
            else:
                enum = path
		
            img_vol = torch.zeros(1, len(tf.io.gfile.listdir(enum)), self.target_size[0], self.target_size[1])
            seg_vol = torch.zeros(1, len(tf.io.gfile.listdir(enum)), self.target_size[0], self.target_size[1])
		
            for x, f in enumerate(tf.io.gfile.listdir(enum)):
                full_img_path = path +  "full_CT_images/" + f if self.list[i].startswith("P") else path + f
                full_seg_path = path + "masks/" + f if self.list[i].startswith("P") else None
                #img = np.array(Image.open(full_img_path))
                with tf.io.gfile.GFile(full_img_path, 'rb') as png_file:
                    png_bytes = png_file.read()
                    img = tf.image.decode_image(png_bytes)
                    img = img.numpy()


                if full_seg_path != None and os.path.exists(full_seg_path):
                    #seg = np.array(Image.open(full_seg_path))
                    with tf.io.gfile.GFile(full_seg_path, 'rb') as png_file:
                        png_bytes = png_file.read()
                        seg = tf.image.decode_image(png_bytes)
                        seg = seg.numpy() 
                else:
                    seg = np.zeros((512,512))

                if img.min() > 0:
                    img -= img.min()
        
                #img, seg = self.augmentations(img.astype(np.uint32), seg.astype(np.uint8))

                img, seg = self._transform(img, seg)
                img = img.squeeze()
                img_vol[0][x] = img
                seg_vol[0][x] = seg
	   
            for y in range(16-len(path)%16):
                img_vol[0][len(path) + y] = torch.zeros(512,512)
                seg_vol[0][len(path) + y] = torch.zeros(512,512)
            
            mu = img_vol.float().mean()
            sigma = img_vol.float().std()
            img_vol = (img_vol - mu)/sigma
	    
            return img_vol, seg_vol

    def _transform(self, img, mask):
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        return img, mask


if __name__ == '__main__':

    DATA_DIR = 'gs://vector-data-bucket-smh/C_Spine_Hackathon'
    dataset = CTData(DATA_DIR, augmentations=None)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for idx, (img, mask) in enumerate(dloader):
        
        print(mask.shape, img.shape, mask.max(), mask.min(), img.max(), img.min())
