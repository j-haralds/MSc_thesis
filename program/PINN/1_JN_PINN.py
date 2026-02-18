import sys
import time
from sklearn.discriminant_analysis import StandardScaler
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('..')

import importlib


# =========================
# Load SAME data as FNO
# =========================

BASE_DIR = Path.cwd()
I = np.load(BASE_DIR.parent / 'program/NO/data_NO/I.npz')['arr_0']   # [N,1,1000]
V = np.load(BASE_DIR.parent / 'program/NO/data_NO/V.npz')['arr_0']   # [N,1,1000]
test_size = 0.1
split = int(test_size * len(I))

I_test  = I[:split]
V_test  = V[:split]
I_train = I[split:]
V_train = V[split:]

# =========================
# Normalize (like sklearn StandardScaler)
# =========================

# reshape to (samples, 1000)
I_train = I_train.reshape(I_train.shape[0], -1)
V_train = V_train.reshape(V_train.shape[0], -1)
I_test  = I_test.reshape(I_test.shape[0], -1)
V_test  = V_test.reshape(V_test.shape[0], -1)

# Normalize (from train set only)
scaler_I = StandardScaler()
scaler_V = StandardScaler()
I_train = scaler_I.fit_transform(I_train)
V_train = scaler_V.fit_transform(V_train)
I_test = scaler_I.transform(I_test)
V_test = scaler_V.transform(V_test)


# Convert to PyTorch tensors
train_dataset = TensorDataset(
    torch.tensor(I_train, dtype=torch.float32),
    torch.tensor(V_train, dtype=torch.float32)
)

loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =========================
# Simple MLP (instead of FNO)
# =========================

model = nn.Sequential(
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 1000)
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.HuberLoss()

# =========================
# Training loop (same structure as FNO)
# =========================

def training(epochs=50, tol=2.5e-4):
    history = {'loss': []}
    
    for epoch in range(epochs):

        for u_batch, y_batch in loader:
            # reshape: [B,1,1000] → [B,1000]
            u_batch = u_batch.squeeze(1)
            y_batch = y_batch.squeeze(1)

            y_pred = model(u_batch)

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        

        history['loss'].append(loss.item())

        if loss.item() < tol:
            print(f'Epoch {epoch+1} loss: {loss.item():.5f} - Early stopping')
            break

        print(f'Epoch {epoch+1} loss: {loss.item():.5f}')

    return model, history


model, history = training(epochs=50)

# =========================
# Test on real PyBaMM test data
# =========================

test = 1
i = I_test[test].reshape(1, -1)  # [1,1000]
v = V_test[test].reshape(1, -1)  # [1,1000]

i_tensor = torch.tensor(i, dtype=torch.float32)

with torch.no_grad():
    y_pred = model(i_tensor)

# Inverse transform to original scale
i = scaler_I.inverse_transform(i)
v = scaler_V.inverse_transform(v)
y_pred = scaler_V.inverse_transform(y_pred.numpy())

# Plot
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,4))

ax1.plot(i.ravel(), label='Input $I(t)$', color='black')
ax2.plot(v.ravel(), label='True $V(t)$', color='tab:red', linestyle='dashed')
ax2.plot(y_pred.ravel(), label='Predicted $V(t)$', color='tab:blue')

ax2.legend()
ax1.legend()
ax2.set_xlabel('Time step')
ax1.set_ylabel('Current [A]')
ax2.set_ylabel('Voltage [V]')

ax3.plot(history['loss'])
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')  
plt.tight_layout()
plt.show()