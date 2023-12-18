import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.models as models
import numpy as np

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom CIFAR-10 dataset class with 1% sampling per class
class CustomCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.cifar_dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform

        # Dictionary to store indices for each class
        self.class_indices = {class_label: [] for class_label in range(10)}

        # Populate the dictionary with indices for each class
        for idx, (_, label) in enumerate(self.cifar_dataset):
            self.class_indices[label].append(idx)

        # Randomly select 1% of indices for each class
        self.selected_indices = []
        for class_label, indices in self.class_indices.items():
            num_samples = int(len(indices))
            selected_indices = torch.randperm(len(indices))[:num_samples]
            self.selected_indices.extend(indices[i] for i in selected_indices)

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        original_idx = self.selected_indices[idx]
        img, label = self.cifar_dataset[original_idx]
        img = self.transform(img)

        return img, label

# Custom collate function to handle the case where the transform returns PIL images
def custom_collate(batch):
    return batch

# Define the transformation for the dataset
train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor()])
test_transform = transforms.ToTensor()

train_set = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=False)
test_set = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=False)

n_classes = 10
# indices = np.random.choice(len(train_set), 60000, replace=False)
# sampler = SubsetRandomSampler(indices)
# train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True)
# test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# Create DataLoader for training and testing with custom collate function
train_loader = DataLoader(train_set, batch_size=64, drop_last = True)#, sampler = sampler
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

# Define the ResNet18 model and move it to CUDA if available
model = models.resnet18().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}')

    # Testing loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
