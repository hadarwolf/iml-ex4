import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np

np.random.seed(42)
torch.manual_seed(42)


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])            
            self.resnet18 = resnet18()
                
        
        in_features_dim = self.resnet18.fc.in_features        
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                    param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        ### YOUR CODE HERE ###
        pass

def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    ### YOUR CODE HERE ###

def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        ### YOUR CODE HERE ###
        return

# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
model = ResNet18(pretrained=False, probing=False)
# Linear probing
# model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
# model = ResNet18(pretrained=True, probing=False)

transform = model.transform
batch_size = 32
num_of_epochs = 50
learning_rate = 0.0001
path = 'PATH_TO_whicfaceisreal' # For example '/cs/usr/username/whichfaceisreal/'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
### Define the loss function and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
### Train the model

# Train the model
for epoch in range(num_of_epochs):
    # Run a training epoch
    loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
    # Compute the accuracy
    train_acc = compute_accuracy(model, train_loader, device)
    # Compute the validation accuracy
    val_acc = compute_accuracy(model, val_loader, device)
    print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
    # Stopping condition
    ### YOUR CODE HERE ###

# Compute the test accuracy
test_acc = compute_accuracy(model, test_loader, device)


import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# 1. Setup the Feature Extractor (The "Frozen Brain")
# We load the weights but we do NOT put it in our custom class. 
# We use the raw model because we want to modify it directly.
weights = ResNet18_Weights.IMAGENET1K_V1
feature_extractor = resnet18(weights=weights)

# IMPORTANT: Remove the final classification layer.
# Originally, .fc outputs 1000 numbers (for ImageNet classes).
# We replace it with Identity() so it outputs the raw 512 features instead.
feature_extractor.fc = nn.Identity()

feature_extractor.to(device)
feature_extractor.eval() # Freeze completely (stop Dropout/Batch Norm updates)

# 2. Define a Helper to Extract Features
def extract_features_to_numpy(loader, model, device):
    all_features = []
    all_labels = []
    
    print(f"Extracting features from {len(loader.dataset)} images...")
    with torch.no_grad(): # No gradients needed, we are just "reading" the images
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(device)
            
            # Pass image through the network -> Get 512 numbers
            feats = model(imgs)
            
            # Move to CPU and convert to Numpy (Sklearn needs Numpy, not PyTorch Tensors)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Stack all batches into one big table
    return np.concatenate(all_features), np.concatenate(all_labels)

# 3. Extract Features (This might take a minute)
print("--- Processing Training Data ---")
X_train, y_train = extract_features_to_numpy(train_loader, feature_extractor, device)

print("--- Processing Test Data ---")
X_test, y_test = extract_features_to_numpy(test_loader, feature_extractor, device)

# 4. Train Logistic Regression (Sklearn)
print("--- Training Sklearn Model ---")
# We use default parameters as requested. 
# max_iter=1000 is added just to prevent warnings if it needs more time to converge.
clf = LogisticRegression(max_iter=1000) 

# Fit (Train)
clf.fit(X_train, y_train)

# 5. Evaluate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"\nBONUS RESULT - Sklearn Linear Probing:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")