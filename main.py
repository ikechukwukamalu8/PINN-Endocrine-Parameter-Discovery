import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Reproducibility
# =====================================================
torch.manual_seed(42)
np.random.seed(42)

# =====================================================
# True Physiological Parameters
# =====================================================
TRUE_P1 = 0.025
G_B = 80.0

# =====================================================
# Synthetic Clinical Data
# =====================================================
def generate_patient_data():
    t = np.linspace(0, 120, 30)
    glucose = G_B + 170 * np.exp(-TRUE_P1 * t) + np.random.normal(0, 2, size=t.shape)
    return (
        torch.tensor(t, dtype=torch.float32).view(-1, 1),
        torch.tensor(glucose, dtype=torch.float32).view(-1, 1)
    )

t_data, G_data = generate_patient_data()

# =====================================================
# High-Precision PINN Model
# =====================================================
class EndocrinePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Fixed: Reduced output to 1 to ensure structural identifiability
        )
        # Learnable physiological parameter
        self.p1 = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))
        
    def forward(self, t):
        return self.network(t)

model = EndocrinePINN()

# =====================================================
# Optimizer
# =====================================================
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# =====================================================
# Physics Collocation Points
# =====================================================
t_phys = torch.linspace(0, 120, 200, requires_grad=True).view(-1, 1)

# =====================================================
# Training Loop with Adaptive Gradient Balancing
# =====================================================
epochs = 6000
parameter_history = []

print("⏳ Training high-precision PINN with adaptive gradient balancing...")

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 1. Compute Individual Losses
    G_pred = model(t_data)
    loss_data = nn.MSELoss()(G_pred, G_data)
    
    G_phys = model(t_phys)
    dGdt = torch.autograd.grad(
        G_phys, t_phys,
        grad_outputs=torch.ones_like(G_phys),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Fixed: Mechanistic physical constraint without unconstrained latent states
    residual = dGdt + model.p1 * (G_phys - G_B)
    loss_phys = torch.mean(residual ** 2)
    
    # 2. Dynamic Gradient Balancing (Resolves Gradient Pathology)
    with torch.no_grad():
        lambda_p = max(0.5, min(5.0, loss_data.item() / (loss_phys.item() + 1e-5)))
        
    total_loss = loss_data + lambda_p * loss_phys
    
    total_loss.backward()
    optimizer.step()
    
    parameter_history.append(model.p1.item())
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}/{epochs} | Loss: {total_loss.item():.4f} | Estimated p1: {model.p1.item():.5f}")

# =====================================================
# Trajectory Reconstruction Output
# =====================================================
estimated_p1 = model.p1.item()

with torch.no_grad():
    t_test = torch.linspace(0, 120, 300).view(-1, 1)
    G_recon = model(t_test).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(t_data.numpy(), G_data.numpy(), marker='x', color='black', s=60, label='Clinical Observations')
plt.plot(t_test.numpy(), G_recon, color='#1f77b4', linewidth=3, label='PINN Reconstruction')
plt.axhline(G_B, linestyle='--', color='gray', label='Basal Glucose Baseline')
plt.title(f'Parameter Discovery Summary\nTrue p1 = {TRUE_P1:.3f} | Discovered p1 = {estimated_p1:.3f}')
plt.xlabel("Time (minutes)")
plt.ylabel("Glucose (mg/dL)")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig("parameter_discovery.png", dpi=300)
plt.close()

# =====================================================
# Parameter Convergence Output
# =====================================================
plt.figure(figsize=(8, 5))
plt.plot(parameter_history, color='#2ca02c', linewidth=2.5, label='Adaptive PINN Path')
plt.axhline(TRUE_P1, linestyle='--', color='black', label=f'True p1 ({TRUE_P1})')
plt.xlabel("Training Epoch")
plt.ylabel("Estimated Clearance Value (p1)")
plt.title("High-Precision Parameter Convergence Tracking")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig("parameter_convergence.png", dpi=300)
plt.close()
