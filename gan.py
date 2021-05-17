import torch.autograd as autograd
import torch.nn.functional as F
import torch
from torchsummary import summary
import dcgan
import wgan_gp
import wgan_gp_contra
from torchvision import transforms

net_G_models = {
    'dcgan_cnn': dcgan.Generator,
    'dcgan_resnet': dcgan.ResGenerator,
    'wgan_gp_cnn': wgan_gp.Generator,
    'wgan_gp_resnet': wgan_gp.ResGenerator,
    'con_gan': wgan_gp_contra.ResConGenerator
}
net_D_models = {
    'dcgan_cnn': dcgan.Discriminator,
    'dcgan_resnet': dcgan.ResDiscriminator,
    'wgan_gp_cnn': wgan_gp.Discriminator,
    'wgan_gp_resnet': wgan_gp.ResDiscriminator,
    'con_gan': wgan_gp_contra.ResConDiscriminator
}

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


class GAN():
    def __init__(self, args):
        # value set
        self.mode = args.mode
        self.gpu = args.gpu
        self.latent_dim = args.latent_dim
        self.lambda_gp = args.lambda_gp
        self.lambda_dcon = args.lambda_dcon
        self.lambda_gcon = args.lambda_gcon
        self.thelta = args.thelta
        self.use_gcon = args.use_gcon
        self.use_dcon = args.use_dcon
        self.t = args.t

        device = torch.device('cuda' if args.gpu else 'cpu')
        # Initialize generator and discriminator

        self.G = net_G_models[args.model](args.latent_dim).to(device)
        summary(self.G, [(args.latent_dim,)],batch_size=args.batch_size,
                device='cuda' if args.gpu else 'cpu')

        self.D = net_D_models[args.model]().to(device)
        summary(self.D, [(args.channels, args.img_size, args.img_size)],
                batch_size=args.batch_size, device='cuda' if args.gpu else 'cpu')

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.b1, args.b2))

        Lambda = lambda step: 1 - step / args.total_steps

        self.sched_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=Lambda)
        self.sched_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=Lambda)

    def train_G(self, real_imgs):
        noise = torch.randn((real_imgs.shape[0], self.latent_dim), device=real_imgs.device)

        gen_imgs = self.G(z=noise)
        adv_fake, f_fake = self.D(x=gen_imgs)

        if self.use_gcon:
            noise_p = torch.rand(noise.shape).to(noise.device)
            noise_p = self.thelta * torch.rand(1).to(noise.device) \
                      * noise_p / torch.norm(noise_p, p=2, dim=1).reshape(noise_p.shape[0],-1)

            noise_n = torch.rand(noise.shape).to(noise.device)
            noise_n = (torch.rand(1) * (1 - self.thelta) + self.thelta).to(noise.device)\
                      * noise_n / torch.norm(noise_n, p=2, dim=1).reshape(noise_n.shape[0],-1)
            imgs_p = self.G(z=(noise_p+noise).detach())
            imgs_n = self.G(z=(noise_n+noise).detach())

            _, f_p = self.D(x=imgs_p)
            _, f_n = self.D(x=imgs_n)


            contra = torch.exp((f_fake * f_p).sum(dim=1)/self.t)
            gcon_loss = -torch.log2(contra / (contra + torch.exp(
                torch.mm(f_fake, f_n.T)/self.t).sum(dim=1))).mean()

        if self.mode == 'wgan':
            gf_loss = -adv_fake.mean()
        elif self.mode == 'dcgan':
            gf_loss = F.binary_cross_entropy_with_logits(adv_fake, torch.ones_like(adv_fake))
        else:
            return "method Error"

        g_loss = gf_loss + self.lambda_gcon * gcon_loss if self.use_gcon else gf_loss

        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        errG = {
            'g_loss':g_loss.item(),
            'gf_loss':gf_loss.item(),
            'gcon_loss':gcon_loss.item() if self.use_gcon else 0
        }

        return errG

    def train_D(self, real_imgs):

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter

            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp

        noise = torch.randn((real_imgs.shape[0], self.latent_dim), device=real_imgs.device)
        gen_imgs = self.G(z=noise).detach()

        if self.use_dcon:
            trans_imgs = []

            for i in range(real_imgs.shape[0]):
                trans_imgs.append(tf(real_imgs[i]).unsqueeze(0).detach())

            trans_imgs = torch.cat(trans_imgs, dim=0).to(real_imgs.device)
            adv_trans, f_trans = self.D(x=trans_imgs)

        adv_real, f_real = self.D(x=real_imgs)
        adv_fake, _ = self.D(x=gen_imgs)

        if self.mode == 'wgan':
            wd = 0.5 * (adv_real.mean() + adv_trans.mean()) - adv_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, real_imgs, gen_imgs)

        elif self.mode == 'dcgan':
            df_loss = F.binary_cross_entropy_with_logits(adv_real, torch.ones_like(adv_real)) + \
                      F.binary_cross_entropy_with_logits(adv_fake, torch.zeros_like(adv_fake))

        else:
            return "method Error"

        d_loss = df_loss

        if self.mode == 'wgan':
            d_loss = d_loss + self.lambda_gp * df_gp

        if self.use_dcon:
            contra = torch.exp(torch.mm(f_real, f_trans.T) / self.t)

            dcon_loss = -torch.log2((contra * torch.eye(contra.shape[0]).to(contra.device)).sum(dim=1) /
                                    contra.sum(dim=1)).mean()

            d_loss = d_loss + self.lambda_dcon * dcon_loss

        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()

        errD = {
            'd_loss':d_loss.item(), 'df_loss':df_loss.item(),
            'dcon_loss':dcon_loss.item() if self.use_dcon else 0,
            'df_gp':df_gp.item() if self.mode == 'wgan' else 0.0,
        }

        return errD

    def step_G_D(self):
        self.sched_G.step()
        self.sched_D.step()

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'sched_G': self.sched_G.state_dict(),
            'sched_D': self.sched_D.state_dict(),
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)

        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optimizer_G' in states:
            self.optimizer_G.load_state_dict(states['optimizer_G'])
        if 'optimizer_D' in states:
            self.optimizer_D.load_state_dict(states['optimizer_D'])
        if 'sched_G' in states:
            self.sched_G.load_state_dict(states['sched_G'])
        if 'sched_D' in states:
            self.sched_D.load_state_dict(states['sched_D'])

    def save_G(self, path, flag=None):
        states = {
            'G': self.G.state_dict(),
        }
        if flag is None:
            torch.save(states, path)
        elif flag == 'unzip':
            torch.save(states, f=path, _use_new_zipfile_serialization=False)

    def save_D(self, path):
        states = {
            'D': self.G.state_dict()
        }
        torch.save(states, path)

if __name__ == '__main__':
    pass
