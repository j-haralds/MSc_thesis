import sys
import time
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
        loss_sum = 0
        nb = 0
        start_b = time.perf_counter()

        for u_batch, y_batch in loader:
            # reshape: [B,1,1000] → [B,1000]
            u_batch = u_batch.squeeze(1)
            y_batch = y_batch.squeeze(1)

            y_pred = model(u_batch)

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            nb += 1

        end_b = time.perf_counter()
        avg_loss = loss_sum / nb
        ETA = (end_b-start_b)*(epochs-epoch-1)

        history['loss'].append(avg_loss)

        if avg_loss < tol:
            print(f'Epoch {epoch+1} avg loss: {avg_loss:.5f} - Early stopping')
            break

        print(f'Epoch {epoch+1} avg loss: {avg_loss:.5f} ETA: {int(ETA//60)}:{int(ETA%60):02d}')

    return model, history


model, history = training(epochs=50)

# =========================
# Test on real PyBaMM test data
# =========================

test = 0
i = I_test[test]
v = V_test[test]

i_tensor = torch.tensor(i, dtype=torch.float32).squeeze(0).unsqueeze(0)

with torch.no_grad():
    y_pred = model(i_tensor)

# Plot
f, ax = plt.subplots(2,1,figsize=(9,6.5),
                     gridspec_kw={'height_ratios': [.5, 1]},
                     sharex=True)
f.subplots_adjust(hspace=0.05)

ax[0].plot(i.ravel(), label='Input $I(t)$', color='black')
ax[1].plot(v.ravel(), label='True $V(t)$',
           color='tab:red', linestyle='dashed')
ax[1].plot(y_pred.detach().numpy().ravel(),
           label='Predicted $V(t)$',
           color='tab:blue')

ax[1].legend()
ax[0].legend()
ax[1].set_xlabel('Time step')
ax[0].set_ylabel('Current [A]')
ax[1].set_ylabel('Voltage [V]')

plt.figure()
plt.plot(history['loss'])
plt.title("Training Loss")
plt.show()