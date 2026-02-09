import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt

import os
import random

# ------------------------------------------------
# Dataset
# ------------------------------------------------

class DepthDataset(Dataset):
    # Initializes dataset access and indexing
    def __init__(self, image_folder, action_folder, label_folder, transform):
        self.image_folder = image_folder
        self.action_folder = action_folder
        self.label_folder = label_folder
        self.transform = transform

        self.image_files = sorted(os.listdir(image_folder))
        self.action_files = sorted(os.listdir(action_folder))
        self.label_files = sorted(os.listdir(label_folder))

    # Gets total number of data items
    def __len__(self):
        return len(self.image_files)

    # Gets item at index
    def __getitem__(self, idx):
        image_file = os.path.join(self.image_folder, self.image_files[idx])
        action_file = os.path.join(self.action_folder, self.action_files[idx])
        label_file = os.path.join(self.label_folder, self.label_files[idx])

        image = np.load(image_file)
        action = np.load(action_file)
        label = np.load(label_file)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        action = torch.tensor(action, dtype=torch.float32)

        return image, action, label


# ------------------------------------------------
#  FiLM Block
#  (Feature Channels, Action Condition) -> (Modulated Feature Channels)
# ------------------------------------------------

class FiLM(nn.Module): 
    def __init__(self, num_channels, condition_dims):
        super().__init__()

        hidden_dims = 128
        self.mlp = nn.Sequential(
            nn.Linear(condition_dims, hidden_dims)
            , nn.ReLU()
            , nn.Linear(hidden_dims, 2 * num_channels)
        )

        # Initialize final layer identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Initialize gamma = 1, beta = 0
        with torch.no_grad():
            self.mlp[-1].bias[:num_channels].fill_(1.0)

    def forward(self, x, condition):
        '''
        Computes gamma, beta using an MLP
        Returns the input layer, features modulated by: gamma * x + beta
        
        :x: (B, num_ch, H, W)
        :condition: (B, cond_dims)

        :return: (B, num_ch, H, W)
        '''
        modulation_vals = self.mlp(condition)               # modulation_vals: (B, cond_dims) -> (B, 2 * C)
        gamma, beta = modulation_vals.chunk(2, dim=1)       # gamma, beta: (B, 2 * num_ch) -> (B, num_ch), (B, num_ch)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)           # gamma: (B, num_ch) -> (B, num_ch, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)             # beta: (B, num_ch) -> (B, num_ch, 1, 1)

        return gamma * x + beta                         
    

# ------------------------------------------------
#  FiLM Double Conv Block
#  (Feature Channels, Action Condition) -> (Feature Channels)
# ------------------------------------------------

class FiLMDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dims):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film1 = FiLM(out_channels, condition_dims)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.film2 = FiLM(out_channels, condition_dims)
        self.relu2 = nn.ReLU(inplace=True)

    # double convolution block w/ FiLM: two consecutive 3x3 conv + FiLM + ReLU
    def forward(self, x, condition):
        '''
        Computes gamma, beta using an MLP
        Returns the input layer, features modulated by: gamma * x + beta
        
        :x: (B, in_ch, H, W)
        :condition: (B, cond_dims)

        :return: (B, out_ch, H, W)
        '''
        x = self.conv1(x)                                   # x: (B, in_ch, H, W) -> (B, out_ch, H, W)
        x = self.film1(x, condition)                        # x: (B, out_ch, H, W), (B, cond_dims) -> (B, out_ch, H, W)
        x = self.relu1(x)                                   # x: (B, out_ch, H, W) -> (B, out_ch, H, W)

        x = self.conv2(x)                                   # x: (B, out_ch, H, W) -> (B, out_ch, H, W)
        x = self.film2(x, condition)                        # x: (B, out_ch, H, W), (B, cond_dims) -> (B, out_ch, H, W)
        x = self.relu2(x)                                   # x: (B, out_ch, H, W) -> (B, out_ch, H, W)

        return x


# ------------------------------------------------
#  Double Conv Block
#  (Feature Channels, Action Condition) -> (Feature Channels)
# ------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    # double convolution block: two consecutive 3x3 conv + ReLU
    def forward(self, x):
        x = self.conv1(x)                                   # x: (B, in_ch, H, W) -> (B, out_ch, H, W)
        x = self.relu1(x)                                   # x: (B, out_ch, H, W) -> (B, out_ch, H, W)

        x = self.conv2(x)                                   # x: (B, out_ch, H, W) -> (B, out_ch, H, W)
        x = self.relu2(x)                                   # x: (B, out_ch, H, W) -> (B, out_ch, H, W)

        return x

# ------------------------------------------------
#  UNet Model
#  (Prior Environment, Action) -> (Next Environment)
# ------------------------------------------------

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], condition_dims=128):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # --- Encoder ---
        in_ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_ch, feature))
            self.downs.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_ch = feature

        # --- Bottleneck ---
        self.bottleneck = FiLMDoubleConv(in_ch, in_ch * 2, condition_dims)
        in_ch = in_ch * 2

        # --- Decoder ---
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_ch, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

            in_ch = feature

        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x, condition):
        skip_scale = 0.1
        skip_connections = []

        # --- Encoder ---
        for idx in range(0, len(self.downs), 2):
            conv = self.downs[idx]
            down = self.downs[idx + 1]

            # Increase channels
            x = conv(x)

            # Store skip connection
            skip_connections.append(x * skip_scale)

            # Decrease spatial dimensions
            x = down(x)

        # --- Bottleneck ---
        x = self.bottleneck(x, condition)

        # Reverse skip connections
        skip_connections = skip_connections[::-1]

        # --- Decoder ---
        for idx in range(0, len(self.ups), 2):
            up = self.ups[idx]
            conv = self.ups[idx + 1]
            
            # Increase spatial dimensions, decrease channels for skip connection concatenation
            x = up(x)

            # Concatenate skip connection (increases channels)
            skip_connection = skip_connections[idx // 2]
            x = torch.cat((skip_connection, x), dim=1)

            # Decrease channels
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
def convert_depth_to_png(tensor):
    tensor = tensor.clone()
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor

# Generates num_sample random samples from the dataset, runs them through the model, and saves input, prediction, and label side by side
def save_samples(epoch, model, dataset, device, prefix, num_samples=10, folder='output', residual_scale=100):
    model.eval()
    os.makedirs(f'{folder}/{prefix}_epoch_{epoch + 1}', exist_ok=True)

    # Random indices for sampling
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, action, label = dataset[idx]
            # Adds batch dimension, moves data to device
            image_batch = image.unsqueeze(0).to(device)
            action_batch = action.unsqueeze(0).to(device)
            label_batch = label.unsqueeze(0).to(device)

            # Feeds input images through forward pass to get predicted residual
            pred_diff_batch = model(image_batch, action_batch) / residual_scale
            # Calculates predicted image
            pred_image_batch = image_batch + pred_diff_batch
            # Cacluates the expected residual
            exp_diff_batch = label_batch - image_batch

            # Removes batch dimension, moves data to CPU
            image_np = image_batch.squeeze(0).cpu()
            label_np = label_batch.squeeze(0).cpu()
            exp_diff_np = exp_diff_batch.squeeze(0).cpu()
            pred_diff_np = pred_diff_batch.squeeze(0).cpu()
            pred_image_np = pred_image_batch.squeeze(0).cpu()

            # Denormalizes data
            image_d  = denormalize(image_np).clamp(0, 1)
            label_d  = denormalize(label_np).clamp(0, 1)
            exp_diff_d   = denormalize(exp_diff_np).clamp(0, 1)
            pred_diff_d   = denormalize(pred_diff_np).clamp(0, 1)
            pred_image_d   = denormalize(pred_image_np).clamp(0, 1)

            # Transforms data for visualization as a png
            image_vis = convert_depth_to_png(image_d)
            label_vis = convert_depth_to_png(label_d)
            exp_diff_vis = convert_depth_to_png(exp_diff_d)
            pred_diff_vis = convert_depth_to_png(pred_diff_d)
            pred_image_vis = convert_depth_to_png(pred_image_d)

            # Saves images side by side
            images_to_save = torch.stack([image_vis, label_vis, exp_diff_vis, pred_diff_vis, pred_image_vis], dim=0) 
            save_path = f'{folder}/{prefix}_epoch_{epoch + 1}/sample_{i}.png'

            # Creates grid and saves images with nrow images per row -> (input, pred, label)
            save_image(images_to_save, save_path, nrow=5)

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
    BATCH_SIZE = 4
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    RESIDUAL_SCALE = 100

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
    train_dataset_folder = 'inputs'
    action_dataset_folder = 'actions'
    label_dataset_folder = 'labels'

    dataset = DepthDataset(train_dataset_folder, action_dataset_folder, label_dataset_folder, data_transforms)
    train_size = int(0.8 * len(dataset))
    validation_size = int(0.0 * len(dataset))
    test_size = len(dataset) - (train_size + validation_size)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = UNet(in_channels=1, out_channels=1, condition_dims=21).to(device)

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

        for images, actions, labels in train_dataloader:
            images = images.to(device)
            actions = actions.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            delta_pred = model(images, actions)
            delta_scaled = (labels - images) * RESIDUAL_SCALE
            loss = criterion(delta_pred, delta_scaled)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(epoch_train_loss)

        # ------------------------------------------------
        # Evaluation
        # ------------------------------------------------
        # n/a

        # ------------------------------------------------
        # Testing
        # ------------------------------------------------
        model.eval()
        running_test_loss = 0.0

        with torch.no_grad():
            for images, actions, labels in test_dataloader:
                images = images.to(device)
                actions = actions.to(device)
                labels = labels.to(device)

                delta_pred = model(images, actions)
                delta_scaled = (labels - images) * RESIDUAL_SCALE
                loss = criterion(delta_pred, delta_scaled)

                running_test_loss += loss.item()

        epoch_test_loss = running_test_loss / len(test_dataloader)
        test_losses.append(epoch_test_loss)

        print(f'-- Epoch {epoch + 1} / {NUM_EPOCHS} --')
        print(f'Train Loss: {epoch_train_loss}')
        print(f'Test Loss: {epoch_test_loss}')

        # ------------------------------------------------
        # Generate sample images
        # ------------------------------------------------
        # Generates 16 samples for train every 2 epochs
        if (epoch + 1) % 10 == 0:
            # Generate samples from training dataset
            save_samples(epoch, model, train_dataset, device, prefix='train', num_samples=16, folder='output', residual_scale=RESIDUAL_SCALE)

            # Generates samples from test dataset if test set is non-empty
            if len(test_dataset) > 0:
                save_samples(epoch, model, test_dataset, device, prefix='test', num_samples=4, folder='output', residual_scale=RESIDUAL_SCALE)

    # After training, plot the losses
    plot_loss(train_losses, test_losses)