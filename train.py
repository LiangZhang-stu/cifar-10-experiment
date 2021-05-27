import argparse
import os
import json
import datetime

from multiprocessing import cpu_count
from tqdm import tqdm
from score.both import get_inception_score_and_fid
from gan import GAN
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets
from os.path import join
from helpers import add_scalar_dict
from tqdm._tqdm import trange
from tensorboardX import SummaryWriter
import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def infiniteloop(dataloader):
    while True:
        for x, _ in iter(dataloader):
            yield x


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--total_steps", type=int, default=50000, help="number of epochs of training")
    parser.add_argument("--n_d", type=int, default=5, help="# of d updates per g update")
    parser.add_argument("--imgs_num", type=int, default=50000, help="length of dataset")
    parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
    parser.add_argument("--test_size", dest="test_size", type=int, default=50, help="size of the test_batches")
    parser.add_argument("--eval_step", dest="eval_step", type=int, default=2500,
                        help="epoch of the formulate FID and IS")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--num_workers", dest='num_workers', type=int, default=cpu_count(),
    #                     help="number of cpu threads to use during batch generation")
    parser.add_argument("--num_workers", dest='num_workers', type=int, default=cpu_count(),
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=bool, default=False, help="whether to use gpu")

    parser.add_argument("--use_gcon", dest="use_gcon", type=bool, default=True, help="whether to use gcon")
    parser.add_argument("--use_dcon", dest="use_dcon", type=bool, default=True, help="whether to use dcon")


    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
    parser.add_argument("--save_interval", type=int, default=2000, help="interval between weight save")
    parser.add_argument("--mode", type=str, default='dcgan')
    parser.add_argument("--lambda_gp", dest='lambda_gp', type=float, default=10.0)

    parser.add_argument("--lambda_gcon", dest='lambda_gcon', type=float, default=1.0)
    parser.add_argument("--lambda_dcon", dest='lambda_dcon', type=float, default=1.0)
    parser.add_argument("--thelta", dest='thelta', type=float, default=0.05)
    parser.add_argument("--t", dest='t', type=float, default=0.5)

    parser.add_argument('--is_resume', dest='is_resume', type=bool, default=False)
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--event_name', dest='event_name', type=str, default=None)
    parser.add_argument('--load_iter', dest='load_iter', type=int, default=0)
    parser.add_argument('--model', dest='model', type=str, default='con_gan')
    parser.add_argument('--data_save_root', dest='data_save_root', type=str, default='')
    parser.add_argument('--ms_file_name', dest='ms_file_name', type=str, default='')
    parser.add_argument('--data_path', dest='data_path', type=str, default='')
    parser.add_argument('--weight_path', dest='weight_path', type=str, default='')
    parser.add_argument("--experiment_name", dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I-%M%p on %B %d_%Y"))
    return parser.parse_args()


args = parse()
print(args)

# make dirs
os.makedirs(join(args.data_save_root, args.experiment_name), exist_ok=True)
os.makedirs(join(args.data_save_root, args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join(args.data_save_root, args.experiment_name, 'sample_training'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'summary'), exist_ok=True)
ms_file_name = join('m1s1_np.npz')

with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

datapath = join(args.data_path, 'cifar')
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root=datapath, train=True, download=False, transform=trans)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=False, num_workers=args.num_workers)
looper = infiniteloop(train_dataloader)

print('Training images:', len(train_dataset))
set_seed(args.seed)

# cudnn.benchmark = True
print('load model')
gan = GAN(args)
if args.is_resume:
    gan.load(os.path.join('output', args.experiment_name, 'checkpoint', 'weights.' + str(args.load_iter) + '.pth'))
    step = args.load_iter
else:
    step = 0

print('learning rate:{}'.format(gan.optimizer_G.param_groups[0]['lr']))
print("load_iteration:{}".format(step))

noise = torch.randn((args.n_samples ** 2, args.latent_dim))
fixed_noise = noise.cuda() if args.gpu else noise

writer = SummaryWriter(join(args.data_save_root, args.experiment_name, 'summary'))
device = torch.device('cuda' if args.gpu else 'cpu')

with trange(step, args.total_steps, dynamic_ncols=True) as pbar:
    for it in pbar:
        writer.add_scalar('LR/learning_rate', gan.optimizer_G.param_groups[0]['lr'], it + 1)
        gan.train()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(args.n_d):
            imgs = next(looper)
            other_imgs = next(looper)
            # Configure input
            imgs = imgs.to(device)
            other_imgs = other_imgs.to(device)
            errD = gan.train_D(real_imgs=imgs, other_imgs=other_imgs)
        add_scalar_dict(writer, errD, it + 1, 'D')

        # -----------------
        #  Train Generator
        # -----------------
        errG = gan.train_G(real_imgs=imgs)
        add_scalar_dict(writer, errG, it + 1, 'G')
        pbar.set_postfix(iter=it + 1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])

        if (it + 1) % args.save_interval == 0:
            gan.save(os.path.join(
                args.data_save_root, args.experiment_name,
                'checkpoint', 'weights.{:d}.pth'.format(it)
            ))

        if (it + 1) % args.sample_interval == 0:
            gan.eval()
            with torch.no_grad():
                samples = gan.G(z=fixed_noise)
                save_image(
                    samples, join(args.data_save_root, args.experiment_name,
                                  'sample_training', 'It:{:d}.jpg'.format(it + 1)),
                    nrow=round(args.n_samples), normalize=True, range=(-1., 1.))

        # set learning rate
        gan.step_G_D()


        # test generate image's is score and fid score
        if (it + 1) % args.eval_step == 0:
            gan.eval()
            fake_imgs = []
            assert args.imgs_num % args.test_size == 0, print('args.imgs_num % args.test_size != 0')

            gbar = tqdm(
                total=args.imgs_num, dynamic_ncols=True, leave=False,
                disable=False, desc="Generate Fake Images")

            for _ in range(args.imgs_num // args.test_size):
                noise = torch.randn((args.test_size, args.latent_dim)).to(device)
                with torch.no_grad():
                    gen_imgs = gan.G(z=noise)
                    gen_imgs = (gen_imgs - gen_imgs.min()) / (gen_imgs.max() - gen_imgs.min())
                    fake_imgs.append(gen_imgs.cpu())

                gbar.update(args.test_size)

            fake_imgs = torch.cat(fake_imgs, dim=0)
            IS, FID = get_inception_score_and_fid(fake_imgs, ms_file_name, verbose=True, weight_path=args.weight_path)

            writer.add_scalar('Score/IS_mean', IS[0], it + 1)
            writer.add_scalar('Score/IS_std', IS[1], it + 1)
            writer.add_scalar('Score/FID', FID, it + 1)

            pbar.write(
                "%s/%s Inception Score: %.3f(%.5f), FID Score: %6.3f"
                % (it, args.total_steps, IS[0], IS[1], FID))

    writer.close()
