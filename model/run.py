'''
python run.py --mask-kspace --data-split val --checkpoint checkpoints/best_model.pt --out-dir 4x/ --challenge singlecoil --data-path ../../singlecoil/ --accelerations 4 --center-fractions 0.08
'''

import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.args import Args
from common.subsample import MaskFunc
from common.utils import save_reconstructions
from data import transforms
from data.mri_data import SliceData
from unet import UNetIntegration5

class DataTransform:
    """
    Data Transformer for running U-Net models on a test dataset.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, kspace_x=368):
        """
        Args:
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.mask_func = mask_func
        self.kspace_x = kspace_x

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.Array): k-space measurements
            target (numpy.Array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object
            fname (pathlib.Path): Path to the input file
            slice (int): Serial number of the slice
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Normalized zero-filled input image
                mean (float): Mean of the zero-filled image
                std (float): Standard deviation of the zero-filled image
                fname (pathlib.Path): Path to the input file
                slice (int): Serial number of the slice
        """
        kspace = transforms.to_tensor(kspace)
        if self.mask_func is not None:
            seed = tuple(map(ord, fname))
            masked_kspace, _ = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image)
        image = image.clamp(-6, 6)

        # difference between kspace actual and target dim
        extra = int(masked_kspace.shape[1] - self.kspace_x)

        # clip kspace at input dim
        if extra > 0:
            masked_kspace = masked_kspace[:, (extra//2):-(extra//2), :]

        # zero pad if necessary
        elif extra < 0:
            empty_kspace = torch.zeros((masked_kspace.shape[0], self.kspace_x, masked_kspace.shape[2]))
            empty_kspace[:, -(extra//2):(extra//2), :] = masked_kspace
            masked_kspace = empty_kspace

        #TODO return mask as well for exclusive updates
        return masked_kspace, image, mean, std, fname, slice


def create_data_loaders(args):
    mask_func = None
    if args.mask_kspace:
        mask_func = MaskFunc(args.center_fractions, args.accelerations)
    data = SliceData(
        root=args.data_path / f'{args.challenge}_{args.data_split}',
        transform=DataTransform(args.resolution, args.challenge, mask_func),
        sample_rate=1.,
        challenge=args.challenge
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return data_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = UNetIntegration5().to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (kspace, image, mean, std, fnames, slices) in data_loader:
            image = image.unsqueeze(1).to(args.device)
            kspace = kspace.permute(0,3,1,2).to(args.device)

            recons, out_k, s_recon = model(kspace, image)
            recons, s_recon = recons.to('cpu').squeeze(1), s_recon.squeeze(1)

            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():
    parser = Args()
    parser.add_argument('--mask-kspace', action='store_true',
                        help='Whether to apply a mask (set to True for val data and False '
                             'for test data')
    parser.add_argument('--data-split', choices=['val', 'test'], required=True,
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=8, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
