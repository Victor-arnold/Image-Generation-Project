# 训练和生成

pytorch环境下，定位到code文件夹

在命令行中运行

```
pip install stylegan2_pytorch
```

再运行

```
stylegan2_pytorch --data path --image-size 32 --name stylegan --aug-prob 0.25 --network-capacity 128 --num-train-steps 40000 --batch_size 5 --gradient_accumulate_every 6
```

path是cifar10_horse所在文件夹

训练完后，可以通过下列命令生成图片

```
python generate.py
```

# 注意事项

该网络比较大，可能会出现GPU显存不够等现象，可以修改batch_size或者增加虚拟内存。

batch_size减小的同时需要增加gradient-accumulate-every