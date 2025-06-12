
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from mambapy.mamba import Mamba, MambaConfig


class SlotAttention(nn.Module):
    """
    Slot Attention Module: Dynamically learns feature representations.
    """
    def __init__(self, num_slots, slot_dim, iters=3):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, num_slots, slot_dim))
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )
        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs):
        B, N, D = inputs.size()
        slots = self.slots_mu + torch.randn_like(self.slots_sigma) * self.slots_sigma
        slots = slots.expand(B, -1, -1)

        inputs = self.norm_input(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)
            k = self.project_k(inputs)
            v = self.project_v(inputs)

            attn = torch.softmax(q @ k.transpose(-2, -1) / D**0.5, dim=-1)
            updates = attn @ v

            slots = self.gru(updates.view(-1, D), slots_prev.view(-1, D))
            slots = slots.view(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots.mean(dim=1)


class TransformerMambaHybridWithSlotAttention(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=8, n_layers=4, num_blocks=3, num_slots=4, slot_dim=64, dropout=0.2):
        super(TransformerMambaHybridWithSlotAttention, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.mamba_blocks = nn.ModuleList([
            Mamba(MambaConfig(d_model=d_model, n_layers=2))
            for _ in range(num_blocks)
        ])
        self.slot_attention = SlotAttention(num_slots=num_slots, slot_dim=slot_dim)
        self.fc_out = nn.Linear(slot_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_layer(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x.unsqueeze(1)).squeeze(1)
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x.unsqueeze(1)).squeeze(1)
        x = self.slot_attention(x.unsqueeze(1))  # Slot Attention
        x = self.dropout(x)
        return self.fc_out(x)


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, weight_decay=1e-4, grad_accum_steps=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

    best_r2 = float('-inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.cuda.amp.autocast():
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0 or i == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss, y_actual, y_pred = 0.0, [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                predictions = model(X_val)
                val_loss += criterion(predictions, y_val).item()
                y_pred.extend(predictions.cpu().numpy().flatten())
                y_actual.extend(y_val.cpu().numpy().flatten())

        val_loss /= len(val_loader)
        mse = mean_squared_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_model_state = model.state_dict()

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Val MSE = {mse:.4f}, R² = {r2:.4f}")

    return best_model_state, best_r2


# Load and preprocess data
train_data_path = "/path/to/training_data.csv"
test_data_path = "/path/to/testing_data.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

X_train = torch.tensor(train_data.drop(columns=['yield']).values, dtype=torch.float32)
y_train = torch.tensor(train_data['yield'].values, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(test_data.drop(columns=['yield']).values, dtype=torch.float32)
y_test = torch.tensor(test_data['yield'].values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize and train the model
model = TransformerMambaHybridWithSlotAttention(input_dim=X_train.shape[1])
best_model_state, best_r2 = train_model(model, train_loader, val_loader, epochs=100)

# Save the best model
torch.save(best_model_state, "best_transformer_mamba_slot_attention_model.pth")
print(f"Best model saved with R²: {best_r2:.4f}")



from sklearn.preprocessing import StandardScaler

# Normalize the training and test data
scaler = StandardScaler()

# Fit on training data and transform
X_train_scaled = torch.tensor(scaler.fit_transform(train_data.drop(columns=['yield']).values), dtype=torch.float32)
y_train = torch.tensor(train_data['yield'].values, dtype=torch.float32).view(-1, 1)

# Transform test data using the same scaler
X_test_scaled = torch.tensor(scaler.transform(test_data.drop(columns=['yield']).values), dtype=torch.float32)
y_test = torch.tensor(test_data['yield'].values, dtype=torch.float32).view(-1, 1)

# Use the scaled data for the DataLoader
train_dataset = TensorDataset(X_train_scaled, y_train)
test_dataset = TensorDataset(X_test_scaled, y_test)

# Split train dataset into training and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def test_on_test_data(test_data_path, model_path, output_graph_path):
    # Load the test dataset
    testing_data = pd.read_csv(test_data_path)
    X_test = testing_data.drop(columns=['yield'])
    y_test = testing_data['yield']

    # Convert test data to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)


    # Define the model architecture 
    model = TransformerMambaHybridWithSlotAttention(input_dim=X_test.shape[1])

    # Load the checkpoint and extract the model state_dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move the model to the appropriate device
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Generate predictions on test data
    y_pred = []
    y_actual = y_test_tensor.cpu().numpy().flatten()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
        y_pred.extend(predictions.flatten())

    # Calculate metrics
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    # Plot predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, alpha=0.6, color='blue', edgecolor='black', s=20)
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)],
             color='red', linestyle='--', linewidth=1.5)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title(f"Test Data Prediction\n$R^2 = {r2:.2f}$, MSE = {mse:.2f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_graph_path, dpi=300)
    plt.close()

    print(f"Evaluation on test data complete. Predictions plotted and saved to {output_graph_path}.")
    print(f"Metrics: R² = {r2:.4f}, MSE = {mse:.4f}")



# Define paths 
test_data_path = '/home/ibrahim/Desktop/dpird/testing_data22.csv'
model_path = 'best_transformer_mamba_slot_attention_model.pth'
output_graph_path = 'improved_model_test_data_prediction_with_attention.png'

# Run the test function on test data
test_on_test_data(test_data_path, model_path, output_graph_path)
