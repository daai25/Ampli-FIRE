"""
CIFAR-100 Image Classification with Convolutional Neural Network (15 Classes)

This script implements a CNN for classifying images from the CIFAR-100 dataset.
CIFAR-100 contains 60,000 32x32 color images in 100 classes, grouped into 15 categories.
The 100 fine classes are mapped to 15 meaningful superclasses for easier classification.

The network architecture includes:
- 2 convolutional layers with ReLU activation and max pooling
- 3 fully connected layers for classification
- Cross-entropy loss and SGD optimizer
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

#import kagglehub
# Download latest version (currently commented out)
#path = kagglehub.dataset_download("pranked03/urbansound8k-mel-spectrogram-images")
#print("Path to dataset files:", path)


def main():
    """
    Main function that implements the complete CIFAR-100 training and evaluation pipeline.
    
    Steps:
    1. Load and preprocess CIFAR-100 dataset
    2. Map 100 fine classes to 15 custom categories
    3. Create and train a CNN model
    4. Evaluate model performance on test set
    5. Display per-class accuracy results
    """
    print("CIFAR-100 15-Class Classification")

    # DATA PREPROCESSING
    # Define image transformations: convert to tensor and normalize pixel values
    # Normalization: (pixel - 0.5) / 0.5 transforms [0,1] range to [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor and scale to [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1,1]
    ])

    # Set batch size for training (number of images processed together)
    batch_size = 4

    # CIFAR-100 CLASS MAPPING
    # Map CIFAR-100's 100 fine classes to 15 music genre categories
    # Based on musical instruments, equipment, and cultural elements
    fine_to_custom_mapping = {
        # Rock (0) - Amplified instruments, powerful equipment
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0,  # powerful sound waves
        50: 0, 51: 0,  # electronic devices → amplifiers, speakers
        
        # Pop (1) - Mainstream production, accessible formats
        5: 1, 6: 1, 7: 1, 8: 1, 9: 1,  # flowing melodies
        35: 1, 36: 1,  # popular, accessible content
        
        # Hip-Hop & Rap (2) - Urban culture, street performance
        60: 2, 61: 2, 62: 2, 63: 2, 64: 2,  # street culture mobility
        85: 2, 86: 2,  # buildings → urban venues
        
        # Electronic (3) - Synthesizers, digital production
        52: 3, 53: 3, 54: 3, 40: 3, 41: 3,  # electronic/tech items → synths, digital gear
        87: 3, 88: 3,  # infrastructure → digital networks
        
        # R&B & Soul (4) - Smooth vocals, emotional expression
        90: 4, 91: 4, 92: 4, 93: 4, 94: 4,  # landscapes → emotional depth
        70: 4, 71: 4,  # trees → organic, soulful growth
        
        # Jazz (5) - Sophisticated improvisation, live performance
        80: 5, 81: 5, 82: 5, 83: 5, 84: 5,  # people → live performers
        75: 5, 76: 5,  # flowers → sophisticated arrangements
        
        # Classical (6) - Orchestral instruments, formal venues
        15: 6, 16: 6, 17: 6, 18: 6, 19: 6,  # grand orchestral sound
        45: 6, 46: 6,  # concert hall seating
        
        # Country & Folk (7) - Acoustic instruments, rural themes
        10: 7, 11: 7, 12: 7, 13: 7, 14: 7,  # rural storytelling themes
        65: 7, 66: 7,  # acoustic instruments and equipment
        
        # Latin (8) - Percussion, rhythmic instruments
        25: 8, 26: 8, 27: 8, 28: 8, 29: 8,  # rhythmic percussion elements
        30: 8, 31: 8,  # hand percussion instruments
        
        # Metal (9) - Heavy guitars, intense sound
        22: 9, 23: 9, 24: 9,  # aggressive sound intensity
        67: 9, 68: 9, 69: 9,  # heavy machinery → distorted guitars
        55: 9,  # additional heavy sonic element
        
        # Punk & Hardcore (10) - DIY instruments, raw sound
        32: 10, 33: 10, 34: 10,  # underground scene aesthetics
        95: 10, 96: 10, 97: 10,  # DIY fashion and culture
        98: 10,  # additional rebellious element
        
        # Reggae & Ska (11) - Caribbean rhythms, island instruments
        37: 11, 38: 11, 39: 11,  # tropical island culture
        72: 11, 73: 11, 74: 11,  # Caribbean natural setting
        99: 11,  # additional tropical element
        
        # World & International (12) - Global instruments, cultural diversity
        42: 12, 43: 12, 44: 12,  # cultural exchange vessels
        89: 12, 56: 12, 57: 12,  # global connectivity infrastructure
        20: 12,  # additional global cultural element
        
        # Blues (13) - Emotional instruments, intimate venues
        47: 13, 48: 13, 49: 13,  # intimate blues venue furniture
        77: 13, 78: 13, 79: 13,  # emotional beauty expressions
        21: 13,  # additional soulful element
        
        # Other (14) - Experimental sounds, avant-garde
        58: 14, 59: 14,  # experimental sonic elements
    }

    # Our 15 custom class names (based on music genre groupings)
    classes = ('Rock', 'Pop', 'Hip-Hop & Rap', 'Electronic', 'R&B & Soul', 
               'Jazz', 'Classical', 'Country & Folk', 'Latin', 'Metal',
               'Punk & Hardcore', 'Reggae & Ska', 'World & International', 'Blues', 'Other')

    def map_labels(labels):
        """
        Map CIFAR-100 fine labels to our 15 custom categories.
        
        Args:
            labels: Tensor of CIFAR-100 fine class labels
            
        Returns:
            Tensor of mapped labels for 15 categories
        """
        mapped_labels = torch.zeros_like(labels)
        for i, label in enumerate(labels):
            mapped_labels[i] = fine_to_custom_mapping[label.item()]
        return mapped_labels

    # Custom Dataset wrapper to handle label mapping
    class MappedCIFAR100(torchvision.datasets.CIFAR100):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            # Map the fine label to our custom 15-class system
            target = fine_to_custom_mapping[target]
            return img, target

    # DATASET LOADING
    # Load CIFAR-100 training dataset with custom mapping
    trainset = MappedCIFAR100(root='./data', train=True,
                              download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # Load CIFAR-100 test dataset with custom mapping
    testset = MappedCIFAR100(root='./data', train=False,
                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    def imshow(img):
        """
        Display a tensor image by converting it back to numpy format.
        
        Args:
            img: PyTorch tensor representing an image
        """
        img = img / 2 + 0.5     # Unnormalize: convert from [-1,1] back to [0,1]
        npimg = img.numpy()     # Convert tensor to numpy array
        # Transpose from (channels, height, width) to (height, width, channels)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # SAMPLE DATA VISUALIZATION
    # Get a batch of training images to visualize
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Display sample images in a grid
    imshow(torchvision.utils.make_grid(images))
    
    # Print the class names for the displayed images
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # NEURAL NETWORK ARCHITECTURE
    class Net(nn.Module):
        """
        Convolutional Neural Network for CIFAR-100 15-category classification.
        
        Architecture:
        - Conv Layer 1: 3→6 channels, 5x5 kernel, ReLU activation, 2x2 max pooling
        - Conv Layer 2: 6→16 channels, 5x5 kernel, ReLU activation, 2x2 max pooling
        - FC Layer 1: 400→120 neurons, ReLU activation
        - FC Layer 2: 120→84 neurons, ReLU activation  
        - FC Layer 3: 84→15 neurons (output layer for 15 classes)
        """
        def __init__(self):
            super(Net, self).__init__()
            # First convolutional layer: 3 input channels (RGB), 6 output channels, 5x5 kernel
            self.conv1 = nn.Conv2d(3, 6, 5)
            # Max pooling layer: 2x2 pooling window with stride 2
            self.pool = nn.MaxPool2d(2, 2)
            # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
            self.conv2 = nn.Conv2d(6, 16, 5)
            # Fully connected layers
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 channels * 5x5 spatial dimensions = 400 inputs
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 15)  # 15 outputs for 15 classes

        def forward(self, x):
            """
            Forward pass through the network.
            
            Args:
                x: Input tensor of shape (batch_size, 3, 32, 32)
                
            Returns:
                Output tensor of shape (batch_size, 15) with class scores
            """
            # First conv block: Conv → ReLU → MaxPool
            # Input: (batch_size, 3, 32, 32) → Output: (batch_size, 6, 14, 14)
            x = self.pool(F.relu(self.conv1(x)))
            
            # Second conv block: Conv → ReLU → MaxPool
            # Input: (batch_size, 6, 14, 14) → Output: (batch_size, 16, 5, 5)
            x = self.pool(F.relu(self.conv2(x)))
            
            # Flatten for fully connected layers
            # Input: (batch_size, 16, 5, 5) → Output: (batch_size, 400)
            x = torch.flatten(x, 1)
            
            # Fully connected layers with ReLU activation
            x = F.relu(self.fc1(x))  # (batch_size, 400) → (batch_size, 120)
            x = F.relu(self.fc2(x))  # (batch_size, 120) → (batch_size, 84)
            x = self.fc3(x)          # (batch_size, 84) → (batch_size, 15)
            return x

    # MODEL INITIALIZATION
    net = Net()
    print(f"Network architecture:\n{net}")

    # TRAINING SETUP
    # Loss function: Cross-entropy loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Optimizer: Stochastic Gradient Descent with momentum
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # TRAINING LOOP
    print("Starting training...")
    for epoch in range(2):  # Train for 2 epochs (complete passes through dataset)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get batch of training data
            inputs, labels = data

            # Zero the gradients from previous iteration
            optimizer.zero_grad()

            # Forward pass: compute predictions
            outputs = net(inputs)
            
            # Compute loss between predictions and true labels
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters
            optimizer.step()

            # Print statistics every 2000 mini-batches
            running_loss += loss.item()
            if i % 2000 == 1999:    # Print every 2000 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # SAVE TRAINED MODEL
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

    # TESTING ON SAMPLE IMAGES
    print("\nTesting on sample images...")
    
    # Get a batch of test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Display test images
    print("Sample test images:")
    imshow(torchvision.utils.make_grid(images))
    print('Ground Truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # Load the trained model for inference
    net = Net()
    net.load_state_dict(torch.load(PATH))

    # Make predictions on sample images
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)  # Get class with highest probability
    print('Predicted:    ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    # OVERALL ACCURACY EVALUATION
    print("\nEvaluating overall accuracy...")
    
    correct = 0  # Number of correct predictions
    total = 0    # Total number of test samples
    
    # Disable gradient computation for faster inference
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            # Get predicted class (index with maximum score)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # Add batch size to total
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Calculate and display overall accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10,000 test images: {accuracy:.1f}%')
    print(f'Correct predictions: {correct}/{total}')

    # PER-CLASS ACCURACY EVALUATION
    print("\nEvaluating per-class accuracy...")
    
    # Initialize counters for each class
    correct_pred = {classname: 0 for classname in classes}  # Correct predictions per class
    total_pred = {classname: 0 for classname in classes}    # Total samples per class

    # Evaluate predictions for each class
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            
            # Count correct and total predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Calculate and display accuracy for each class
    print("\nPer-class accuracy results:")
    print("-" * 35)
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for {classname:>5s}: {accuracy:5.1f}% ({correct_count}/{total_pred[classname]})')
    print("-" * 35)

if __name__ == '__main__':
    # Enable multiprocessing support for Windows
    multiprocessing.freeze_support()
    main()
