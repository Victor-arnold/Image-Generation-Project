### 更改文件路径 进行训练和评分

在__main__中更改cifar10_horse文件路径即可运行\
文件目录下有 5000 张32*32的马图片

```python
data_dir = r"E:/py_data/cifar10_horse/train"  # 这里需要手动更改
```

为了保证运行效率

这个代码只会在达到最大设定训练轮数时进行生成1000张图片 并进行Fid评估

同时，我们会在每一个epoch 保存生成器和判别器的模型参数到 checkpoints 文件夹中，方便对所有轮数中的模型进行评估

如果需要对文件夹中所有生成器的生成图片并评分，需要手动更改下面代码中的路径 并设定 cal_all = True

```python
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
```

### 更改训练模型 和 参数

为了便于更改调整，我们将需要更改的参数以config的方式记录下来  可以根据需要进行修改  

注：WGAN-GP 还存在着部分问题未来得及调整，所以虽然他可以运行，但是无法生成马的图片

```python
config = {
    "model_type": "DCGAN",  # "DCGAN" "WGAN" "WGAN-GP"
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
```





