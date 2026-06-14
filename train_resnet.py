
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# GPU на Mac через Metal
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Устройство: {device}")

# Трансформации изображений
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Датасет — папки Train/Ripe, Train/Unripe, Train/Overripe
train_data = datasets.ImageFolder("dataset/archive-3/Train", transform=train_transform)
test_data  = datasets.ImageFolder("dataset/archive-3/Test",  transform=test_transform)

print(f"Train: {len(train_data)} фото, классы: {train_data.classes}")
print(f"Test:  {len(test_data)} фото")

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=0)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=32, shuffle=False, num_workers=0)

# ResNet-50 с предобученными весами
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Замораживаем все слои кроме последнего
for p in resnet.parameters():
    p.requires_grad = False

# Заменяем последний слой под наши 3 класса
num_classes = len(train_data.classes)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=1e-3)

def evaluate():
    resnet.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total

print("\n--- Фаза 1: обучаем только последний слой (5 эпох) ---")
for epoch in range(5):
    resnet.train()
    correct = total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    val_acc = evaluate()
    print(f"Epoch {epoch+1}/5 | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}%")

print("\n--- Фаза 2: размораживаем последний блок (5 эпох) ---")
for p in resnet.layer4.parameters():
    p.requires_grad = True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-4)

for epoch in range(5):
    resnet.train()
    correct = total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    val_acc = evaluate()
    print(f"Epoch {epoch+1}/5 | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}%")

# Сохраняем модель
torch.save({
    "model_state": resnet.state_dict(),
    "classes": train_data.classes,
    "num_classes": num_classes
}, "resnet_ripeness.pth")
print("\nМодель сохранена: resnet_ripeness.pth")
print(f"Классы: {train_data.classes}")
