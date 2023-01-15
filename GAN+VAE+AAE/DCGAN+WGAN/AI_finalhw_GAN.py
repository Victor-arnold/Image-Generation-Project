import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from datetime import datetime
import os
import glob
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from fid_score.fid_score import FidScore


def set_seed(seed):
    random.seed(seed)  # Random
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # Torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    '''设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。但是由于噪声和不同的硬件条件，
    即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，只需要disable它即可。 '''
    cudnn.deterministic = True
    '''CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。为了避免这种情况，
    就要将这个flag设置为True，让它使用确定的实现。'''


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class MyDataset(Dataset):
    def __init__(self, filenames, transform):
        self.transform = transform
        self.filenames = filenames
        self.num = len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num


def get_dataset(dir):
    fnames = glob.glob(os.path.join(dir, '*'))

    tfm_ = [
        transforms.ToPILImage(),
        # transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    tfm = transforms.Compose(tfm_)
    dataset = MyDataset(fnames, tfm)

    return dataset


class Generator(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True),
            )

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU(),
        )

        self.layer2to5 = nn.Sequential(
            dconv_bn_relu(feature_dim * 8, feature_dim * 4),  # (batch, feature_dim * 16, 8, 8)
            dconv_bn_relu(feature_dim * 4, feature_dim * 2),  # (batch, feature_dim * 16, 16, 16)
            dconv_bn_relu(feature_dim * 2, feature_dim),  # (batch, feature_dim * 16, 32, 32)
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.layer1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.layer2to5(y)
        return y


class Discriminator(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator, self).__init__()

        # 注：如果是WGAN-GP 使用nn.InstanceNorm2d 而不是nn.Batchnorm
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        # 注：如果是WGAN  则要移除Sigmoid层
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=5, stride=1, padding=2),  # (batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(feature_dim, feature_dim * 2),  # (batch, 3, 16, 16)
            conv_bn_lrelu(feature_dim * 2, feature_dim * 4),  # (batch, 3, 8, 8)
            conv_bn_lrelu(feature_dim * 4, feature_dim * 8),  # (batch, 3, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(-1)
        return y


class Discriminator_WGAN(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator_WGAN, self).__init__()

        # 注：如果是WGAN-GP 使用nn.InstanceNorm2d 而不是nn.Batchnorm
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        # 注：如果是WGAN  则要移除Sigmoid层
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=5, stride=1, padding=2),  # (batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(feature_dim, feature_dim * 2),  # (batch, 3, 16, 16)
            conv_bn_lrelu(feature_dim * 2, feature_dim * 4),  # (batch, 3, 8, 8)
            conv_bn_lrelu(feature_dim * 4, feature_dim * 8),  # (batch, 3, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            # nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(-1)
        return y


class Discriminator_WGAN_GP(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator_WGAN_GP, self).__init__()

        # 注：如果是WGAN-GP 使用nn.InstanceNorm2d 而不是nn.Batchnorm
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 4, 2, 1),
                nn.InstanceNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        # 注：如果是WGAN  则要移除Sigmoid层
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=5, stride=1, padding=2),  # (batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(feature_dim, feature_dim * 2),  # (batch, 3, 16, 16)
            conv_bn_lrelu(feature_dim * 2, feature_dim * 4),  # (batch, 3, 8, 8)
            conv_bn_lrelu(feature_dim * 4, feature_dim * 8),  # (batch, 3, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(-1)
        return y


class TrainerGAN():
    def __init__(self, config, G, D, loss):
        self.config = config
        self.G = G
        self.D = D
        self.loss = loss

        # 注： GAN 用Adam  WGAN 用RMSprop  WGAN-GP 用Adam
        if self.config["model_type"] == "WGAN":
            self.opt_D = torch.optim.RMSprop(self.D.parameters(), lr=self.config["lr"])
            self.opt_G = torch.optim.RMSprop(self.G.parameters(), lr=self.config["lr"])
        else:
            self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
            self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))

        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')  # 输出日志路径
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')  # 输出模型检查点路径

        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M')

        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).to(self.config["device"])

    def preparation(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 利用时间更新路径
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, time + f'_{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, time + f'_{self.config["model_type"]}')
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)

        # 创建数据集
        dataset = get_dataset(os.path.join(self.config["data_dir"]))

        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True,
                                     num_workers=self.config["num_workers"])

        self.G = self.G.to(self.config["device"])
        self.D = self.D.to(self.config["device"])
        self.G.train()
        self.D.train()

    def gp(self, r_imgs, f_imgs, Lambda=10):

        epsilon = torch.rand(r_imgs.size(0), 1, 1, 1, device=self.config["device"])

        G_interpolation = (epsilon * r_imgs + (1 - epsilon) * f_imgs).requires_grad_(True)
        D_interpolation = self.D(G_interpolation)

        fake = torch.full((r_imgs.size(0),), 1, device=self.config["device"])

        gradients = torch.autograd.grad(outputs=D_interpolation, inputs=G_interpolation,
                                        grad_outputs=fake, only_inputs=True,
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), - 1)
        grad_penalty = Lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return grad_penalty

    def train(self):

        self.preparation()

        for e, epoch in enumerate(range(self.config["n_epoch"])):

            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e + 1}")

            for i, data in enumerate(progress_bar):
                imgs = data.to(self.config["device"])
                batch_size = imgs.size(0)  # batch 大小

                # -----------------------------------训练判别器------------------------------------------------
                z = Variable(torch.randn(batch_size, self.config["z_dim"])).to(self.config["device"])

                r_imgs = Variable(imgs).to(self.config["device"])  # 真实图片
                f_imgs = self.G(z)  # 生成的假图片
                r_label = torch.ones(batch_size).to(self.config["device"])
                f_label = torch.zeros(batch_size).to(self.config["device"])

                # 前向计算
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)

                """
                不同模型计算的损失
                GAN: 
                    loss_D = (r_loss + f_loss)/2
                WGAN: 
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
                WGAN-GP: 
                    gradient_penalty = self.gp(r_imgs, f_imgs)
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                """

                if self.config["model_type"] == "DCGAN":
                    r_loss = self.loss(r_logit, r_label)
                    f_loss = self.loss(f_logit, f_label)
                    loss_D = (r_loss + f_loss) / 2
                elif self.config["model_type"] == "WGAN":
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
                elif self.config["model_type"] == "WGAN-GP":
                    gradient_penalty = self.gp(r_imgs, f_imgs)
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty

                # 反向传播
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()
                # ------------------------------------------------------------------------------------------------

                # WGAN的 clip操作  可以使其训练变得更稳定
                if self.config["model_type"] == "WGAN":
                    for p in self.D.parameters():
                        p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])

                # -------------------------------------------训练生成器----------------------------------------------
                if self.steps % self.config["n_critic"] == 0:
                    # 生成假图片
                    z = Variable(torch.randn(batch_size, self.config["z_dim"])).to(self.config["device"])
                    f_imgs = self.G(z)

                    # 前向计算
                    f_logit = self.D(f_imgs)

                    """
                    不同模型计算的损失
                    GAN: loss_G = self.loss(f_logit, r_label)
                    WGAN: loss_G = -torch.mean(self.D(f_imgs))
                    WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
                    """

                    if self.config["model_type"] == "DCGAN":
                        loss_G = self.loss(f_logit, r_label)
                    elif self.config["model_type"] == "WGAN":
                        loss_G = -torch.mean(self.D(f_imgs))
                    elif self.config["model_type"] == "WGAN-GP":
                        loss_G = -torch.mean(self.D(f_imgs))

                    # 反向传播
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()
                # ------------------------------------------------------------------------------------------------

                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1

            self.G.eval()
            f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.log_dir, f'Epoch_{epoch + 1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'保存了部分图片在 {filename}.')

            # grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(grid_img.permute(1, 2, 0))
            # plt.show()

            self.G.train()

            # 保存模型
            if (e + 1) % self.config["num_epoch_savecheckpoints"] == 0 or e == 0:
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e + 1}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e + 1}.pth'))

        logging.info('训练完成')

    def inference(self, G_path, n_generate=1000, n_output=30, show=False):

        self.G.load_state_dict(torch.load(G_path))
        self.G.to(self.config["device"])
        self.G.eval()
        z = Variable(torch.randn(n_generate, self.config["z_dim"])).to(self.config["device"])
        imgs = (self.G(z).data + 1) / 2.0
        os.makedirs('output', exist_ok=True)
        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], f'output/{i + 1}.jpg')

            # if show:
            #     row, col = n_output // 10 + 1, 10
            # grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
            # plt.figure(figsize=(row, col))
            # plt.imshow(grid_img.permute(1, 2, 0))
            # plt.show()


def fid_score_calculation(r_image_dir, generate_image_dir, batch_size=64):
    paths = [r_image_dir, generate_image_dir]
    batch_size = batch_size
    fid = FidScore(paths, device, batch_size)
    score = fid.calculate_fid_score()
    print("模型最后的Fid分数是:", score)
    return score


if __name__ == "__main__":
    # 设置随机种子
    set_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    workspace_dir = "."  # 当前工作路径

    data_dir = r"E:/py_data/cifar10_horse/train"  # 这里需要手动更改

    config = {
        "model_type": "WGAN",  # "DCGAN" "WGAN" "WGAN-GP"
        "batch_size": 64,
        "lr": 3e-4,
        "n_epoch": 600,
        "n_critic": 3,
        "z_dim": 100,
        "num_workers": 4,
        "workspace_dir": workspace_dir,
        "data_dir": data_dir,
        "device": device,
        "clip_value": 0.01,
        "num_epoch_savecheckpoints": 1,  # 每几轮保存一次模型
    }

    G = Generator(100)

    if config["model_type"] == "DCGAN":
        D = Discriminator(3)
    elif config["model_type"] == "WGAN":
        D = Discriminator_WGAN(3)
    elif config["model_type"] == "WGAN-GP":
        D = Discriminator_WGAN_GP(3)

    loss = nn.BCELoss()

    trainer = TrainerGAN(config=config, G=G, D=D, loss=loss)

    trainer.train()

    G_file_name = config["model_type"]  # 存放checkpoints的文件夹名字
    num_of_G = config["n_epoch"]
    G_path = "./checkpoints/" + G_file_name + "/G_" + str(num_of_G) + ".pth"  # 生成器的路径
    trainer.inference(G_path=G_path)  # 生成图片
    a = fid_score_calculation(r_image_dir=data_dir, generate_image_dir="./output")
    print(f"生成图片的fid为 {a}")

    # -----------------------------------------------------------------------------------------------------------------
    # 计算checkpoint文件夹下面所有的生成器的Fid
    cal_all = False

    G_files_name = "2022-07-06_07-47-05_DCGAN"  # 存放checkpoints的文件夹名字
    num_epoch = 1000  # 根据checkpoints的个数来设定
    fidlist = []
    if cal_all:
        result = 10000
        num = 0
        for i in range(num_epoch):
            # 生成图片
            G_path = "./checkpoints/" + G_files_name + "/G_" + str(i + 1) + ".pth"
            trainer.inference(G_path=G_path)
            # 计算Fid分数
            a = fid_score_calculation(r_image_dir=data_dir, generate_image_dir="./output")
            fidlist.append([i + 1, a])
            if a < result:
                result = a
                num = i + 1

        print(fidlist)
        print(f"最优的结果是第 {num} 轮的生成器，Fid分数为 {result}")
