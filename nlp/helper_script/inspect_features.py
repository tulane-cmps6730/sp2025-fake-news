import torch
import os

# Define the directory where features are saved
data_dir = 'data/preprocessed'

# Load the extracted features
train_features = torch.load(os.path.join(data_dir, 'train_features.pt'))
test_features = torch.load(os.path.join(data_dir, 'test_features.pt'))

# Check the shape of the features
print("Train Features Shape:", train_features.shape)
print("Test Features Shape:", test_features.shape)

# Optionally, inspect a sample of the features
print("\nSample Train Feature:")
print("Shape:", train_features[0].shape)
print("Mean:", train_features[0].mean().item())
print("Std:", train_features[0].std().item())
print("Min:", train_features[0].min().item())
print("Max:", train_features[0].max().item()) 