import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 1. PyTorch Dataset Class
class MaintainabilityDataset(Dataset):
    def __init__(self, csv_file, feature_mask=None):
        """
        Args:
            csv_file (str): Path to the processed dataset.
            feature_mask (list / np.array): Binary array from the Genetic Algorithm 
                                            (e.g., [1, 0, 1...]) to select specific features.
        """
        # Load the data
        self.data_frame = pd.read_csv(csv_file)
        
        # X: All columns except file_name (index 0) and target_bug_proneness (last column)
        self.X = self.data_frame.iloc[:, 1:-1].values.astype(np.float32)
        
        # Apply the feature mask from Genetic Algorithm (if provided)
        if feature_mask is not None:
            mask_indices = [i for i, val in enumerate(feature_mask) if val == 1]
            self.X = self.X[:, mask_indices]
            
        # Y: The Target Bug Proneness (last column)
        self.y = self.data_frame.iloc[:, -1].values.astype(np.float32)
        
        # Convert to PyTorch Tensors
        self.X = torch.tensor(self.X)
        
        # Reshape y from [batch_size] to [batch_size, 1] so it matches network output
        self.y = torch.tensor(self.y).view(-1, 1)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 2. PyTorch Neural Network Architecture
class MaintainabilityANN(nn.Module):
    def __init__(self, input_dim):
        """
        A 3-layer architecture to prevent overfitting while learning complex boundaries.
        """
        super(MaintainabilityANN, self).__init__()
        
        # Hidden Layer 1 (Input -> 32 nodes)
        self.layer1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        
        # Hidden Layer 2 (32 nodes -> 16 nodes)
        self.layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        
        # Output Layer (16 nodes -> 1 continuous prediction)
        # Using Linear because this is an unbounded regression task (predicting bug count)
        self.output_layer = nn.Linear(16, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        
        out = self.layer2(out)
        out = self.relu2(out)
        
        out = self.output_layer(out)
        return out


# 3. Model Training & Evaluation Wrapper
def train_and_evaluate_ann(csv_file, feature_mask=None, epochs=100, batch_size=16):
    """
    Trains the ANN on the given dataset and returns the final validation MSE.
    This will be heavily used by the Genetic Algorithm evaluator.
    """
    # Initialize Dataset and DataLoader for Batching
    dataset = MaintainabilityDataset(csv_file, feature_mask)
    
    # Split into 80% train / 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Automatically detect how many input variables we are using based on the mask
    input_dim = dataset.X.shape[1]
    
    # Initialize Model, Loss Function (MSE), and Optimizer (Adam)
    model = MaintainabilityANN(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            # 1. Forward pass
            predictions = model(batch_X)
            
            # 2. Calculate MSE Loss
            loss = criterion(predictions, batch_y)
            
            # 3. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Evaluation Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()
            
    final_mse = val_loss / len(val_loader)
    return final_mse

if __name__ == "__main__":
    # Test the standalone network with ALL features (using numpy fake-mask of all 1s)
    csv_path = "flask_dataset_processed.csv"
    
    print("Testing PyTorch ANN with all features...")
    # Assuming 14 numeric features
    all_features_mask = [1] * 14 
    score = train_and_evaluate_ann(csv_path, feature_mask=all_features_mask, epochs=50)
    print(f"Final Validation Mean Squared Error (MSE): {score:.4f}")

