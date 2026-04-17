import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from collections import Counter
import pandas as pd

# -------------------------
# PATHS (CHANGE IF NEEDED)
# -------------------------
train_dir = r"C:\Users\Dell\Desktop\soil-app\soil_dataset\train"
test_dir  = r"C:\Users\Dell\Desktop\soil-app\soil_dataset\test"

# -------------------------
# SETTINGS
# -------------------------
batch_size = 32
epochs = 20
lr = 0.0003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# TRANSFORMS
# -------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3,0.3,0.3),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# DATASET
# -------------------------
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data  = datasets.ImageFolder(test_dir, transform=test_transforms)

#WINDOWS FIX
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

class_names = train_data.classes
print("Classes:", class_names)

# -------------------------
# CLASS WEIGHTS
# -------------------------
labels = [label for _, label in train_data]
counts = Counter(labels)
class_weights = [1.0 / counts[i] for i in range(len(counts))]
class_weights = torch.tensor(class_weights).to(device)

# -------------------------
# MODEL
# -------------------------
model = resnet50(weights=ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# -------------------------
# LOSS + OPTIMIZER
# -------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)

# -------------------------
# TRAINING
# -------------------------
best_acc = 0
best_epoch = 0
log_data = []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # -------------------------
    # TEST
    # -------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Loss: {running_loss:.4f}")
    print(f"Train Acc: {train_acc:.2f}%")
    print(f"Test Acc: {test_acc:.2f}%")

    # SAVE LOG
    log_data.append([epoch+1, running_loss, train_acc, test_acc])

    # SAVE BEST MODEL
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_soil_model.pth")
        print("Best model saved")

# -------------------------
# TABLE
# -------------------------
df = pd.DataFrame(log_data, columns=[
    "Epoch", "Loss", "Train Accuracy", "Test Accuracy"
])

print("\n===== TRAINING TABLE =====")
print(df.round(2))

# SAVE CSV
df.to_csv("training_log.csv", index=False)

# -------------------------
# FINAL RESULT
# -------------------------
print("\n===== FINAL RESULT =====")
print(f"Best Test Accuracy: {best_acc:.2f}%")
print(f"Achieved at Epoch: {best_epoch}")

print("\nFiles saved:")
print("- best_soil_model.pth")
print("- training_log.csv")