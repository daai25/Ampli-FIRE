import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class MusicDataset(Dataset):
    def __init__(self, file_list, labels, n_mels=128, duration=15, sr=22050, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.n_mels = n_mels
        self.duration = duration
        self.sr = sr
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        
        # Load audio and convert to mel spectrogram
        y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_db_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Ensure fixed size (n_mels x fixed_time_steps), pad or truncate time dimension
        desired_width =1288  # for example
        if mel_spec_db_norm.shape[1] < desired_width:
            pad_width = desired_width - mel_spec_db_norm.shape[1]
            mel_spec_db_norm = np.pad(mel_spec_db_norm, ((0,0),(0,pad_width)), mode='constant')
        else:
            mel_spec_db_norm = mel_spec_db_norm[:, :desired_width]
        
        # Add channel dimension: (1, n_mels, time)
        mel_spec_db_norm = np.expand_dims(mel_spec_db_norm, axis=0)
        
        mel_tensor = torch.tensor(mel_spec_db_norm, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            mel_tensor = self.transform(mel_tensor)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=8)
        self.time_mask = T.TimeMasking(time_mask_param=20)

        mel_tensor = self.freq_mask(mel_tensor)
        mel_tensor = self.time_mask(mel_tensor)
        
        return mel_tensor, label_tensor


class MusicCNN(nn.Module):
    def __init__(self, num_classes):
        super(MusicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Dummy forward pass to calculate flattened feature size
        self._dummy_input = torch.zeros(1, 1, 128, 1288)  # match your dataset size
        with torch.no_grad():
            x = self._dummy_input
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self.num_features = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.num_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #if batch_idx % 10 == 0:
        #    print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")
    print(f"Epoch {epoch} Average Loss {running_loss / len(train_loader):.4f}")

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return correct / total

def get_files_and_labels(dataset_path):
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))  # folder names sorted alphabetically
    print(class_names)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    for cls_name in class_names:
        cls_folder = os.path.join(dataset_path, cls_name)
        if not os.path.isdir(cls_folder):
            continue
        for file_name in os.listdir(cls_folder):
            if file_name.endswith('.mp3'):  # or other audio extensions
                file_paths.append(os.path.join(cls_folder, file_name))
                labels.append(class_to_idx[cls_name])
    
    return file_paths, labels, class_to_idx


def main():
    file_list, labels, class_to_idx = get_files_and_labels('./audio/')
    print("Classes:", class_to_idx)
    print("Number of files:", len(file_list))

    num_classes = len(class_to_idx)
    batch_size = 4
    epochs = 25

    train_files, test_files, train_labels, test_labels = train_test_split(
        file_list, labels, test_size=0.2, random_state=42)

    train_dataset = MusicDataset(train_files, train_labels)
    test_dataset = MusicDataset(test_files, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MusicCNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        acc = evaluate(model, device, test_loader)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model." + str(acc) + ".mdl")

if __name__ == "__main__":
    main()

