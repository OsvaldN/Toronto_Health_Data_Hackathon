import logging
import pathlib
import random
import shutil
import time

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ct_data import get_loaders

from common.args import Args
from data import transforms
from data.ct_data import SliceData
from unet import UnetModel
from model import deeplabv3_resnet101

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

RATIO = 0.000549764

'''
python train.py --data-path ../train/
'''

def create_datasets(args):

    train_data = SliceData(
        root=args.data_path / f'MICST_train'
    )
    dev_data = SliceData(
        root=args.data_path / f'MICST_val'
    )

    return dev_data, train_data


def create_data_loaders(split='train'):
    train_loader, dev_loader = get_loaders(split=split)
    return train_loader, dev_loader


def train_epoch(args, epoch, model, pos_loader, neg_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(pos_loader)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1/RATIO]).to(args.device))

    for itr, data in enumerate(pos_loader):
        image, mask = data

        n_image, n_mask = next(iter(neg_loader))
        
        mean_loss = 0
        image = image[0][0]
        mask = mask[0][0]
        n_image = n_image[0][0]
        n_mask = n_mask[0][0]

        i=0

        for img,msk in [(image.permute(0,1,2), mask.permute(0,1,2)), (image.permute(1,0,2), mask.permute(1,0,2)), (image.permute(2,0,1), mask.permute(2,0,1)),
                        (n_image.permute(0,1,2), n_mask.permute(0,1,2)), (n_image.permute(1,0,2), n_mask.permute(1,0,2)), (n_image.permute(2,0,1), n_mask.permute(2,0,1))]:
            img = img.unsqueeze(1)
            msk = msk.unsqueeze(1)

            complete = 0
            while complete < img.shape[0]:
                t_img = img[complete:complete + args.batch_size]
                t_msk = msk[complete:complete + args.batch_size]

                t_img, t_msk = t_img.to(args.device), t_msk.to(args.device)
                output = model(t_img)['out']
                loss = criterion(output, t_msk) * torch.Tensor([RATIO]).to(args.device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mean_loss += loss / torch.Tensor([RATIO]).to(args.device)
                i += 1
                complete += args.batch_size
                break
        loss = mean_loss/i

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if itr > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + itr)
        if itr % args.report_interval == 0:

            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{itr:4d}/{len(pos_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dev_loss': 1e9,
                'exp_dir': args.exp_dir
            },
            f=args.exp_dir / 'last_model.pt'
        )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch

def evaluate(args, epoch, model, pos_loader, neg_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for itr, data in enumerate(pos_loader):
            image, mask = data

            n_image, n_mask = next(iter(neg_loader))

            mean_loss = 0
            image = image[0][0]
            mask = mask[0][0]
            n_image = n_image[0][0]
            n_mask = n_mask[0][0]

            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1/RATIO]).to(args.device))

            mean_loss = 0
            image = torch.from_numpy(image[0][0])
            mask = torch.from_numpy(mask[0][0])
            i=0

            for img,msk in [(image.permute(0,1,2), mask.permute(0,1,2)), (image.permute(1,0,2), mask.permute(1,0,2)), (image.permute(2,0,1), mask.permute(2,0,1))]:
                img = img.unsqueeze(1)
                msk = msk.unsqueeze(1)

                complete = 0
                while complete < img.shape[0]:
                    t_img = img[complete:complete + args.batch_size]
                    t_msk = msk[complete:complete + args.batch_size]

                    t_img, t_msk = t_img.to(args.device), t_msk.to(args.device)
                    output = model(t_img)['out']
    
                    loss = criterion(output, t_msk)
                    mean_loss += loss
                    i += 1
                    complete += args.batch_size

            loss = mean_loss/i

            losses.append(loss.item())
            break
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def build_model(args):
    model = deeplabv3_resnet101().to(args.device)
    #model = UnetModel(in_chans=1, out_chans=1, chans=4, num_pool_layers=3, drop_prob=args.drop_prob).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    pos_train_loader, neg_train_loader = create_data_loaders(split='train')
    pos_dev_loader, neg_dev_loader = create_data_loaders(split='dev')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, gamma=args.gamma)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, pos_train_loader, neg_train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, pos_dev_loader, neg_dev_loader, writer)
        #visualize(args, epoch, model, display_loader, writer)
        scheduler.step()

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--drop-prob', type=float, default=0.2, help='Dropout probability')

    parser.add_argument('--batch-size', default=6, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=1,
                        help='Period of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.97,
                        help='Gamma of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--loss', type=str,
                        help='loss criterion one of "MSE", "L1", "SSIM"')

    parser.add_argument('--report-interval', type=int, default=1, help='Period of loss reporting')
    parser.add_argument('--data-parallel', default=False, action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
