import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import time

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
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    

device = torch.device("cpu")

model = Model4().to(device)

model.load_state_dict(torch.load("/home/users/arancic/EuroCC/model4_cifar10.pth", map_location=device))

model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

correct=0
total=0

start_time = time.time()

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

end_time = time.time()
    
print(f"Test accuracy: {100 * correct / total:.2f}%")
print(f"inference time on cpu is: {end_time -start_time:.2f} seconds")

import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os
os.makedirs("/home/users/arancic/EuroCC/inference_outputs", exist_ok=True)

def imshow_save(img_tensor, filename, predicted, true_labels, classes):
    img_tensor = img_tensor / 2 + 0.5
    npimg = img_tensor.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.axis('off')
    ax.set_title(
        "Predicted: " + " | ".join([classes[p.item()] for p in predicted]) + "\n" +
        "Ground Truth: " + " | ".join([classes[l.item()] for l in true_labels])
    )
    plt.tight_layout()
    plt.savefig(os.path.join("/home/users/arancic/EuroCC/inference_outputs", filename))
    plt.close()

# Get a batch
dataiter = iter(testloader)
images, labels = next(dataiter)
images = images.to(device)
labels = labels.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Save first 4 images in a grid with predictions
imshow_save(torchvision.utils.make_grid(images[:4]), "/home/users/arancic/EuroCC/inference_outputs/predictions_batch1.png", predicted[:4], labels[:4], testset.classes)
print("Saved predictions to inference_outputs/predictions_batch1.png")