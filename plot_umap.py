import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from models import SimCLR
import matplotlib.pyplot as plt
import umap
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
test_transform = transforms.ToTensor()
test_set = CIFAR10(root='./data', train=False, transform=test_transform, download=False)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

enc_path = './logs/SimCLR/cifar10/simclr_resnet18_epoch500_batch64.pt'
proj_path = './logs/SimCLR/cifar10/simclr_resnet18_epoch500_batch64.pt'
base_encoder = eval('resnet18')
model = SimCLR(base_encoder, projection_dim=128).cuda()
#pre_model.load_state_dict(torch.load('./logs/SimCLR/cifar10/simclr_{}_epoch{}_batch{}.pt'.format('resnet18', epoch_size, batch_size)))
saved_enc_dict = torch.load(enc_path)
# saved_proj_dict = torch.load(enc_path)
# model.enc.load_state_dict(saved_enc_dict)
# model.proj.load_state_dict(proj_path)
# model = model.to(device)
model.load_state_dict(saved_enc_dict)
label_list = []
projection_list = []
representation_list = []
model.eval()
count = 0
for (inputs, labels) in test_loader:
    if count < 2:
        print(labels.shape)
        label_list.append(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        representation = outputs[0].cpu().detach().numpy()
        projection = outputs[1].cpu().detach().numpy()
        representation_list.append(representation)
        projection_list.append(projection)
        count += 1
representation_list = np.array(representation_list).reshape((-1, 512))
projection_list = np.array(projection_list).reshape((-1, 128))
label_list = np.array(label_list).reshape(-1, 1)
print(representation_list.shape)
print(projection_list.shape)
print(label_list.shape)
        # reducer = umap.UMAP()
        # embedding = reducer.fit_transform(representation)

        # # plt.scatter(
        # #     embedding[:, 0],
        # #     embedding[:, 1],
        # #     c=[sns.color_palette()[x] for x in labels], s=10, rasterized=True)
        # plt.scatter(
        #     embedding[:, 0],
        #     embedding[:, 1],
        #     c=[sns.color_palette()[x] for x in labels], s=10, rasterized=True)
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.axis('off')
        # plt.title('UMAP projection of the Penguin dataset', fontsize=24)
        # plt.savefig(f'./plot/test_plot{count}.svg')
        # plt.savefig(f'./plot/test_plot{count}.pdf')
        # plt.savefig(f'./plot/test_plot{count}.png')
