# 该代码主要分为四个部分：
## 分别是 数据处理、训练、蒸馏、采样生成图片
### 下面分别介绍各个部分的代码使用


# 数据处理
主要运行的文件是prepare_data.py，使用如下命令运行，需要提供数据集文件目录（该目录下要有其类别目录，以datasets.ImageFolder方式读取）同时需要提供图片大小以及输出路径   
`
!python ./prepare_data.py --out ./data/cifar10_horse --size 32 ./cifar10_horse
`
# 训练

训练的代码如下所示  
`
!python ./Main_train.py --module create_model_dataset --name exp_distillation --dname original --batch_size 128 --num_workers 16 --num_iters 600000 --num_timesteps 1024
`  
需要注意的是 如果使用不同的数据集，请仿照cifar10_dataset.py创建一个新的数据集读取文件，同时需要在主要的模块create_model_dataset.py内进行对应的修改

# 蒸馏
蒸馏的代码如下所示  
`!python ./Main_distillate.py --module create_model_dataset --diffusion GaussianDiffusionDefault --name exp_distillation --dname 512_steps --base_checkpoint ./ckpt/exp_distillation/original/checkpoint.pt --batch_size 128 --num_workers 16 --num_iters 5000 --log_interval 5
`

  每运行一次 采样步骤减半 需要指定模型的路径以及保存路径 可以通过多次运行 达到渐进式蒸馏的效果
  
# 采样生成图片
采样生成图片主要有两个形式：  
生成带有扩散过程的图片 代码如下  
`
!python ./Main_sample.py --out_file ./generate_images/origin.png --module create_model_dataset --checkpoint ./ckpt/exp_distillation/original/checkpoint.pt --batch_size 8
`  


生成指定数量的图片 用以计算fid 代码如下  
`
!python ./Main_generate.py --num 50 --out_file ./generate_images/all_images_1024/ --module create_model_dataset --checkpoint ./ckpt/exp_distillation/original/checkpoint.pt 
`






