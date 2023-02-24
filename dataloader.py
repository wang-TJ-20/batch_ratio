import torchvision
from torchvision import datasets
import torch
from torch.utils.data.dataset import ConcatDataset
from samper import BatchSchedulerSampler
from torch.utils.tensorboard import SummaryWriter
import os

# 鲜花数据集的图像预处理
data_transform_flower = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize([32, 32])
    ]
)
# cifar10数据集的图像预处理
data_transform_cifar10 = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

# 鲜花数据集的地址
path='./data/flower'
flower_class = os.listdir(path)


# 生成鲜花数据
flower_data = datasets.ImageFolder(root=path,
                                transform=data_transform_flower)
# 生成cifar10数据
cifar10_data = datasets.CIFAR10('./data', train=True, transform=data_transform_cifar10, download=True)

first_dataset = flower_data
second_dataset = cifar10_data
# 生成组合数据
concat_dataset = ConcatDataset([first_dataset, second_dataset]) # 使用concat_dataset.datasets可以看出这个数据类型中储存了原始的dataset

# 设置一个batch里的数据集比例
ratio = [0.5, 0.5]
batch_size = 8

# dataloader with BatchSchedulerSampler
dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                         sampler=BatchSchedulerSampler(dataset=concat_dataset,
                                                                       batch_size=batch_size,
                                                                       ratio=ratio),
                                         batch_size=batch_size,
                                         shuffle=False)
write = SummaryWriter()
idx = 0
for inputs in dataloader:
    img, target = inputs
    write.add_images('img',img, idx)
    idx += 1
    # print(img.shape)
    # break