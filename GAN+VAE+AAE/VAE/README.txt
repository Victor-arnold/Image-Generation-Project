模型测试了VAE网络的生成与重建效果

模型是建立在pytorch的基础上训练的
除此之外还需要的第三方包有：
tqdm,
torchvision,
pytorch_fid,
glob,
numpy

其中使用了三个文件夹，分别为：
data_path = 'cifar10_horse/train' #训练数据路径
sample_dir = 'Samples' #随机采样生成图像的保存路径
fake_path = 'Reconstruction' #重建图片的保存路径

模型训练使用的网络并没有很复杂，
对于encoder：
Cov:3->16->64
Fcnn:8*8*64->128->h_dim(24)
encoder则相反

默认训练1000个epoch，在RTX2070 SUPER加速下大概需要45min
(实验训练了10000个epoch发现效果并没有改善)
计算出的fid为：
采样：291.1898485349791
重建：167.46181932773797

与GAN相比，VAE的生成效果不是很理想