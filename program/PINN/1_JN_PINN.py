import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

'''
PINN example for simple ODE du/dt + u = 0 with u(0)=1.
The true solution is u(t) = exp(-t).
No data is used for training, only the physics and boundary condition.
'''

# PINN network (simple feedforward)
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    def forward(self, t):
        return self.net(t)

# instantiate
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# time points (collocation)
t = torch.linspace(0, 1, 100).view(-1,1)
t.requires_grad = True

def physics_loss(u, t):
    # compute du/dt
    du_dt = torch.autograd.grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    # PDE residual for du/dt + u = 0
    return torch.mean((du_dt + u)**2)

def boundary_loss(u0):
    # boundary condition at t=0, u(0)=1
    return (u0 - 1.0)**2

def training():
    history = {"loss": [], "phys": [], "bnd": []}

    for epoch in range(5000):
        optimizer.zero_grad()

        u = model(t)               # NN prediction
        phys = physics_loss(u,t)   # PDE residual
        u0 = model(torch.tensor([[0.0]]))  # boundary point
        bnd = boundary_loss(u0)    # boundary loss

        loss = phys + bnd
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())
        history["phys"].append(phys.item())
        history["bnd"].append(bnd.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss {loss.item():.5f} | Phys {phys.item():.5f} | Bnd {bnd.item():.5f}")
    return model, history

model, history = training()

# plot solution
t_plot = torch.linspace(0,1,200).view(-1,1)
with torch.no_grad():
    u_pred = model(t_plot).detach().numpy().ravel()

plt.plot(t_plot.numpy(), u_pred, label="PINN")
plt.plot(t_plot.numpy(), torch.exp(-t_plot).numpy(), '--', label="True")
plt.legend()
plt.show()