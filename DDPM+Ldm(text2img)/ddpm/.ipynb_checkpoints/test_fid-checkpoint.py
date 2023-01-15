from utils import *
import torch
from fid_score.fid_score import FidScore

device = "cuda" if torch.cuda.is_available() else "cpu"

def fid_score_calculation(r_image_dir, generate_image_dir, batch_size=64):
    paths = [r_image_dir, generate_image_dir]
    batch_size = batch_size
    fid = FidScore(paths, device, batch_size)
    score = fid.calculate_fid_score()
    print("模型最后的Fid分数是:", score)
    return score

# 计算fid
data_dir = r"/root/autodl-tmp/ddpm/cifar10_horse/train"
fake_dir = r'/root/autodl-tmp/ddim_fake_images2'

    
a = fid_score_calculation(r_image_dir=data_dir, generate_image_dir=fake_dir)
print(f"生成图片的fid为 {a}")

# # 计算checkpoint文件夹下面所有的生成器的Fid
# cal_all = False

# G_files_name = "2022-12-26_DDPM"  # 存放checkpoints的文件夹名字
# num_epoch = 1000  # 根据checkpoints的个数来设定
# fidlist = []
# if cal_all:
#     result = 10000
#     num = 0
#     for i in range(num_epoch):
#         # 生成图片
#         G_path = "./checkpoints/" + G_files_name + "/G_" + str(i + 1) + ".pth"
#         trainer.inference(G_path=G_path)
#         # 计算Fid分数
#         a = fid_score_calculation(r_image_dir=data_dir, generate_image_dir=fake_dir)
#         fidlist.append([i + 1, a])
#         if a < result:
#             result = a
#             num = i + 1

#     print(fidlist)
#     print(f"最优的结果是第 {num} 轮的生成器，Fid分数为 {result}")

