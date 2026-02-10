import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import AlphabetModel  
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.transpose(1, 2)), 
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = DataLoader(datasets.EMNIST('./data', split='letters', train=True, download=True, transform=transform),
                          batch_size=128, shuffle=True)

model = AlphabetModel(num_classes=26).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"🚀 Training ResNet-18 'Cheat' Model on {device}...")

for epoch in range(3): 
    model.train()
    loop = tqdm(train_loader)
    for images, labels in loop:
        images, labels = images.to(device), (labels - 1).to(device)  

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "emnist_letters_resnet.pth")
print("🔥 Done! High-accuracy model saved.")