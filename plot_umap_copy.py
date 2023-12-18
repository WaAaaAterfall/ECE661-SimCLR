import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from models import SimCLR
import matplotlib.pyplot as plt
import umap
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
#                                           transforms.RandomHorizontalFlip(p=0.5),
#                                           transforms.ToTensor()])
test_transform = transforms.ToTensor()
# test_set = CIFAR10(root='./data', train=False, transform=test_transform, download=False)
# test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

num_images_per_class = 100
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=test_transform)
class_indices = {class_label: np.where(np.array(train_dataset.targets) == class_label)[0] for class_label in range(10)}
selected_indices = []
for class_label in range(10):
    class_indices_for_selection = np.random.choice(class_indices[class_label], num_images_per_class, replace=False)
    selected_indices.extend(class_indices_for_selection)
balanced_dataset = Subset(train_dataset, selected_indices)
batch_size = 1000
balanced_dataloader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)

count = 0
for label in [0.01, 0.1, 0.3, 0.5, 1.0]:#, 500
    for batch in [64]:#, 128, 256
        label_list = []
        projection_list = []
        representation_list = []
        enc_path =f'./logs/SimCLR/cifar10/simclr_finetune_resnet18_label{label}_batch{batch}.pt'
        proj_path = f'./logs/SimCLR/cifar10/simclr_resnet18_epoch500_batch{batch}.pt'
        base_encoder = eval('resnet18')
        model = SimCLR(base_encoder, projection_dim=128).cuda()
        saved_enc_dict = torch.load(enc_path)
        # saved_proj_dict = torch.load(proj_path)
        keys_to_load = ["projector.0.weight", "projector.0.bias", "projector.2.weight", "projector.2.bias"]
        saved_proj_dict = torch.load(proj_path, map_location=torch.device('cpu'))
        prefix_length = len("projector.")
        loaded_keys = {key[prefix_length:]: saved_proj_dict[key] for key in keys_to_load}

        # saved_proj_dict = torch.load(enc_path)
        # model.enc.load_state_dict(saved_enc_dict)
        # model.proj.load_state_dict(proj_path)
        # model = model.to(device)
        model.enc.load_state_dict(saved_enc_dict)
        model.projector.load_state_dict(loaded_keys)
        model.eval()
        for (inputs, labels) in balanced_dataloader:
            label_list.append(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            representation = outputs[0].cpu().detach().numpy()
            projection = outputs[1].cpu().detach().numpy()
            break
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(representation)
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=[sns.color_palette()[x] for x in labels], s=10, rasterized=True)
        plt.gca().set_aspect('equal', 'datalim')
        plt.axis('off')
       # plt.title('UMAP projection', fontsize=24)
        plt.savefig(f'./plot/finetune_batch{batch}_label{label}_representation.svg')
        plt.savefig(f'./plot/finetune_batch{batch}_label{label}_representation.png')
        # plt.savefig(f'./plot/batch{batch}_epoch{epoch}_representation.png')
        # plt.savefig(f'./plot/batch{batch}_epoch{epoch}_representation.svg')
