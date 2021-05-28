# GAN-Cifar10-PyTorch

A graduation design project:
Collected GAN's improved models, and innovation introduces Contrastive Learning to help GAN improve the generated image effect.

This work adopts the evaluation index calculation method of IS and FID in Project(https://github.com/w86763777/pytorch-inception-score-fid)

## Requirements

* Python 3
* matplotlib
* tensorboardX
* torchsummary
* dataclasses
* numpy
* Pillow
* scipy
* torch
* torchvision
* tqdm
* typing-extensions
* tensorboard

```bash
pip install -r requirements.txt
```

* Dataset
  * [Cifar-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset

* [InceptionV3](https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth)
  * if you have download this weight file, you can add it to `data/weight/inceptionv3_weight_name`, 
  * or you can discard `arguments: --inception_weight_path` when you train model(the default value of this arguments is `None`)

* cifar-10 fid stats
```bash
CUDA_VISIBLE_DEVICES=0 \
python calc_stats.py --output=cifar_10_fid_stats.npz --use_torch --data_path=your_cifar-10_dataset_path
```
## Usage


#### To train an DCGAN(CNN) on Cifar10

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py --n_d=1 --gpu=True --data_save_root=output --inception_weight_path=data/weight/inceptionv3_weight_name --ms_file_name=cifar_10_fid_stats.npz --experiment_name=dcgan_cnn --total_steps=100000 --latent_dim=100 --mode='dcgan' --model='dcgan_cnn' --batch_size=128
```

#### To train an WGAN(CNN) on Cifar10

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py --n_d=1 --gpu=True --data_save_root=output --inception_weight_path=data/weight/inceptionv3_weight_name --ms_file_name=cifar_10_fid_stats.npz --experiment_name=wgan_gp_cnn --total_steps=100000 --latent_dim=100 --mode='wgan' --model='wgan_gp_cnn' --batch_size=128	--b1=0 --b2=0.9
```

#### To train an WGAN(Resnet) on Cifar10

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py --n_d=5 --gpu=True --data_save_root=output --inception_weight_path=data/weight/pt_inception-2015-12-05-6726825d.pth --ms_file_name=cifar_10_fid_stats.npz --experiment_name=wgan_gp_resnet --total_steps=100000 --latent_dim=128 --mode='wgan' --model='wgan_gp_resnet' --batch_size=64 --b1=0 --b2=0.9
```

#### To train an WGAN(Resnet) with Contrastive Loss on Cifar10

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py --n_d=5 --gpu=True --data_save_root=output --weight_path=/root/wbw/zhangl/dataset/InceptionV3/pt_inception-2015-12-05-6726825d.pth --data_path=/root/wbw/zhangl/dataset/cifar10 --ms_file_name=m1s1_np.npz --experiment_name=con_gan_5_27 --total_steps=100000 --latent_dim=128 --mode='wgan' --model='con_gan' --batch_size=64 --b1=0 --b2=0.9 --t=0.5
```