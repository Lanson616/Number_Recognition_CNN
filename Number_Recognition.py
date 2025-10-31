import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import struct
import matplotlib.pyplot as plt
import numpy as np

def load_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows*cols)
    return images

def load_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load data
train_images = load_idx_images("train-images.idx3-ubyte") / 255.0
train_labels = load_idx_labels("train-labels.idx1-ubyte")

test_images  = load_idx_images("t10k-images.idx3-ubyte") / 255.0
test_labels  = load_idx_labels("t10k-labels.idx1-ubyte")

# 2. PyTorch Dataset
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.X = torch.tensor(images, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MNISTDataset(train_images, train_labels)
test_dataset = MNISTDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*14*14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16*14*14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

# 4. Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):  # just 3 epochs to start
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Test phase
model.eval()
all_images, all_preds, all_labels = [], [], []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Store results for visualization
        all_images.append(images)
        all_preds.append(predicted)
        all_labels.append(labels)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 6. Visualize some predictions
images = all_images[0]
preds = all_preds[0]
labels = all_labels[0]

plt.figure(figsize=(8,4))
for i in range(6):  # show 6 images
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][0], cmap="gray")
    plt.title(f"Pred: {preds[i].item()} | True: {labels[i].item()}")
    plt.axis("off")
plt.show()