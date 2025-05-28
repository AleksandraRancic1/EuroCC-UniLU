"""
Exercise3: we will train one CNN model using Distributed Data Parallel (DDP) PyTorch.
Dataset - CIFAR100 (much complex than CIFAR10).
Dataloading will be done using 8 CPU cores; I/O are reserved for CPU nodes;
After loading, dataset should be transferred to GPU memory.

Training the model using 2 GPU cores on 1 node:
1. Two clones of the model:
    - PyTorch makes one copy of the model on each GPU
    - These copies start with the same weights and stay in sync.
2. Training data is split:
    - For a batch of 128 images:
    * GPU 0 sees the first 64 images
    * GPU 1 sees the next 64
    - This split is automatic with DistributedSampler
3. Each GPU trains independently:
    - each GPU runs forward pass, compute loss, does backpropagation
4. PyTorch syncs the gradients
    - After backpropagation, PyTorch:
    * averages the gradients from both GOUs
    * makes sure both model copies stay identical
5. Each GPU updates the model
    - Now that gradients are the same:
    * each GPU updates its own copy
    * this results in synchronized training, like using 1 big GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import time
import os

class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),

            nn.Linear(128, 100)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
def main(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device=torch.device(f'cuda:{rank}')
    print(f"[Rank {rank}] Using device: {device}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False)

    trainloader = DataLoader(trainset, batch_size=64, sampler=train_sampler, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size=64, sampler=test_sampler, num_workers=8, pin_memory=True)

    model=Model4().to(device)
    model=DDP(model, device_ids=[rank])

    criterion=nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs=50
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Rank {rank}] Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.4f}")

    end_time = time.time()
    print(f"[Rank {rank}] Training time: {end_time - start_time:.2f} seconds")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    assert world_size == 2, "This script is configured for 2 GPUs"
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
