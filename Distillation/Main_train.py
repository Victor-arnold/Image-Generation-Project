#!/usr/bin/env python
# coding: utf-8
import argparse
import importlib
from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from train_utils import *


def make_argument_parser():
    parser = argparse.ArgumentParser()

    # 必填参数  --module 为 create_model_dataset
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.",
                        type=str, required=True)
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.",
                        type=str, required=True)

    # 其他为选填参数
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--num_timesteps", help="Num diffusion steps.", type=int, default=1024)
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=100000)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=64)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=5e-5)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyConstantLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=15)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=2) # 每两分钟保存一次模型
    parser.add_argument("--num_workers", type=int, default=-1)
    return parser


def train_model(args, make_model, make_dataset):
    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    # 打印参数
    # print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    device = torch.device("cuda")

    
    # 创建数据集  InfinityDataset 在train_util.py里面 第一个参数是数据集 第二个参数是数据集长度
    # make_dataset() 是自己定义的创建数据集的方式
    train_dataset = InfinityDataset(make_dataset(), args.num_iters)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    #test_dataset = InfinityDataset(make_dataset(), args.num_iters)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)

    # 保存点的路径
    #checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)
    
    # 这里改了一下路径的名字
    checkpoints_dir = os.path.join("ckpt", args.name, args.dname)
    
    
    
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    def make_sheduler():
        M = importlib.import_module("train_utils")
        D = getattr(M, args.scheduler)  # 默认是 StrategyConstantLR # 实际上应该是从strategies.py里面调用的
        return D()

    scheduler = make_sheduler()

    def make_diffusion(model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("v_diffusion")
        D = getattr(M, args.diffusion)
        return D(model, betas, time_scale=time_scale)

    # 这里创建了两个Unet
    teacher = make_model().to(device)
    teacher_ema = make_model().to(device)

    # 继续训练与否
    if args.checkpoint_to_continue != "":
        ckpt = torch.load(args.checkpoint_to_continue)
        teacher.load_state_dict(ckpt["G"])
        teacher_ema.load_state_dict(ckpt["G"])
        del ckpt
        print("Continue training...")
    else:
        print("Training new model...")

    # 初始化 ema模型
    init_ema_model(teacher, teacher_ema)

    tensorboard = SummaryWriter(os.path.join(checkpoints_dir, "tensorboard"))

    # 两个diffusion
    teacher_diffusion = make_diffusion(teacher, args.num_timesteps, 1, device)
    teacher_ema_diffusion = make_diffusion(teacher, args.num_timesteps, 1, device)

    
    image_size = teacher.image_size
    # 召回函数  在每一个iter调用
    on_iter = make_iter_callback(teacher_ema_diffusion, device, checkpoints_dir, image_size, tensorboard,
                                 args.log_interval, args.ckpt_interval, False)

    # 正常的diffusion训练 创建对象 和 执行训练 在train_utils.py里面定义
    diffusion_train = DiffusionTrain(scheduler)
    diffusion_train.train(train_loader, teacher_diffusion, teacher_ema, args.lr, device, make_extra_args=make_condition,
                          on_iter=on_iter)
    print("Finished.")


if __name__ == "__main__":
    # 设置参数
    parser = make_argument_parser()
    args = parser.parse_args()

    # 相当于导入模块的作用
    M = importlib.import_module(args.module)
    # 默认情况下导入的是celeba_u.py下的两个函数对象
    make_model = getattr(M, "make_model")  # 这个函数用于返回一个对象的属性值
    make_dataset = getattr(M, "make_dataset")

    # 把三个参数输入 train_model 里面进行训练
    train_model(args, make_model, make_dataset)
