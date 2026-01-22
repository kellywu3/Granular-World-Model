import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image

# ------------------------------------------------
# Dataset
# ------------------------------------------------

# Converts index using mapping
def get_indices_from_mapping(idx, mapping):
    offset = 0
    for file_idx, num_items in enumerate(mapping):
        if offset <= idx and idx < offset + num_items:
            item_idx = idx - offset
            return file_idx, item_idx
    
        offset = offset + num_items

    raise IndexError(f'{__name__}: index {idx} out of range')

class DepthDataset(Dataset):
    # Initializes dataset access and indexing
    def __init__(self, image_folder, label_folder, transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform

        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(label_folder))

        # Assumes image_files and label_files are the same length
        self.timestep_mapping = []
        for file in self.image_files:
            data = np.load(os.path.join(self.image_folder, file), allow_pickle=True)
            payload = data.item()

            self.timestep_mapping.append(len(payload['depth']))

        self.total_timesteps = sum(self.timestep_mapping)

    # Gets total number of data items
    def __len__(self):
        return self.total_timesteps

    # Gets item at index
    def __getitem__(self, idx):
        file_idx, time_idx = get_indices_from_mapping(idx, self.timestep_mapping)

        image_file = os.path.join(self.image_folder, self.image_files[file_idx])
        label_file = os.path.join(self.label_folder, self.label_files[file_idx])

        image_data = np.load(image_file, allow_pickle=True)
        image_payload = image_data.item()
        image = image_payload['depth'][time_idx]

        label_data = np.load(label_file, allow_pickle=True)
        label_payload = label_data.item()
        label = label_payload['depth'][time_idx]

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


# ------------------------------------------------
#  UNet Model
# ------------------------------------------------

# Returns a torch.nn.Sequential double convolution block: two consecutive 3x3 conv + ReLU
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Encoder ---
        in_ch = in_channels
        for feature in features:
            self.downs.append(double_conv(in_ch, feature))
            in_ch = feature

        # --- Bottleneck ---
        self.bottleneck = double_conv(features[-1], features[-1] * 2)

        # --- Decoder ---
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.ups.append(double_conv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # --- Encoder ---
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # Reverse skip connections
        skip_connections = skip_connections[::-1]

        # --- Decoder ---
        for idx in range(0, len(self.ups), 2):
            up = self.ups[idx]
            conv = self.ups[idx + 1]
            # Upsample
            x = up(x)

            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                skip_connection = skip_connection[:, :, diffY // 2 : skip_connection.size()[2] - diffY // 2,
                                                  diffX // 2 : skip_connection.size()[3] - diffX // 2]

            x = torch.cat((skip_connection, x), dim=1)
            x = conv(x)

        x = self.final_conv(x)
        return x


# ------------------------------------------------
# Main
# ------------------------------------------------

# Denormalizes depth data from [-1, 1] to [0, 1]
def denormalize(tensor):
    return (tensor + 1.0) / 2.0

# Transforms tensor for visualizing depth
def depth_to_png(tensor):
    tensor = tensor.clone()
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor

# Generates num_sample random samples from the dataset, runs them through the model, and saves input, prediction, and label side by side
def save_samples(epoch, model, dataset, device, prefix, num_samples=10, folder='output'):
    model.eval()
    os.makedirs(f'{folder}/{prefix}_epoch_{epoch + 1}', exist_ok=True)

    # Random indices for sampling
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            # Adds batch dimension and moves data to device
            image_batch = image.unsqueeze(0).to(device)
            label_batch = label.unsqueeze(0).to(device)

            # Feeds input images through forward pass
            pred_batch = model(image_batch)

            # Detatches batch dimension and moves data to CPU
            image_np = image_batch.squeeze(0).cpu()
            pred_np = pred_batch.squeeze(0).cpu()
            label_np = label_batch.squeeze(0).cpu()

            # Denormalizes data
            image_d  = denormalize(image_np).clamp(0, 1)
            pred_d   = denormalize(pred_np).clamp(0, 1)
            label_d  = denormalize(label_np).clamp(0, 1)

            # Transforms data for visualization as a png
            image_vis = depth_to_png(image_d)
            pred_vis = depth_to_png(pred_d)
            label_vis = depth_to_png(label_d)

            # Saves images side by side
            images_to_save = torch.stack([image_vis, pred_vis, label_vis], dim=0) 
            save_path = f'{folder}/{prefix}_epoch_{epoch + 1}/sample_{i}.png'

            # Creates grid and saves images with nrow images per row -> (input, pred, label)
            save_image(images_to_save, save_path, nrow=3)

# Plots the losses for each epoch
def plot_loss(train_loss, test_loss):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # Changes y-axis to logarithmic scale
    plt.yscale('log')  
    
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    # Hyperparameters
    IMG_SIZE = 64
    BATCH_SIZE = 1
    NUM_EPOCHS = 60
    LEARNING_RATE = 1e-4

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Transforms
    data_transforms = transforms.Compose([
        # Transforms ndarray to tensor
        transforms.ToTensor(),
        # Resizes tensor
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Map 16 bit depth [0,65535] -> [-1,1]
        transforms.Lambda(lambda t: (t / 65535.0) * 2.0 - 1.0),
    ])

    # Datasets
    torch.manual_seed(0)
    train_dataset_folder = 'preliminary_data'
    label_dataset_folder = 'preliminary_data'

    dataset = DepthDataset(train_dataset_folder, label_dataset_folder, data_transforms)
    train_size = int(0.01 * len(dataset))
    validation_size = int(0.985 * len(dataset))
    test_size = len(dataset) - (train_size + validation_size)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = UNet(in_channels=1, out_channels=1).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_max_errors = []

    test_losses = []
    test_max_errors = []

    # ------------------------------------------------
    # Training Loop
    # ------------------------------------------------
    print(f'Running training with {NUM_EPOCHS} epochs and test size {len(train_dataloader)}')
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0
        train_max_error = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            error = (outputs - labels).abs()
            train_max_error = max(train_max_error, error.max().item())

        epoch_train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(epoch_train_loss)
        train_max_errors.append(train_max_error)

        # ------------------------------------------------
        # Evaluation
        # ------------------------------------------------
        # n/a

        # ------------------------------------------------
        # Testing
        # ------------------------------------------------
        model.eval()
        running_test_loss = 0.0
        test_max_error = 0.0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()
                error = (outputs - labels).abs()
                test_max_error = max(test_max_error, error.max().item())

        epoch_test_loss = running_test_loss / len(test_dataloader)
        test_losses.append(epoch_test_loss)
        test_max_errors

        print(f'-- Epoch {epoch + 1} / {NUM_EPOCHS} --')
        print(f'Train Loss: {epoch_train_loss}')
        print(f'Max Absolute Train Error: {train_max_error}')
        print(f'Test Loss: {epoch_test_loss}')
        print(f'Max Absolute Test Error: {test_max_error}')

        # ------------------------------------------------
        # Generate sample images
        # ------------------------------------------------
        # Generates 10 samples for train every 2 epochs
        if (epoch + 1) % 2 == 0:
            # Generate samples from training dataset
            save_samples(epoch, model, train_dataset, device, prefix='train', num_samples=29, folder='output')

            # Generates samples from test dataset if test set is non-empty
            if len(test_dataset) > 0:
                save_samples(epoch, model, test_dataset, device, prefix='test', num_samples=14, folder='output')

    # After training, plot the losses
    plot_loss(train_losses, test_losses)