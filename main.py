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
# True Physiological Parameter
# =====================================================

TRUE_P1 = 0.025
G_B = 80.0

# =====================================================
# Synthetic Clinical Data
# =====================================================

def generate_patient_data():

    t = np.linspace(0,120,30)

    glucose = (
        G_B
        +
        170*np.exp(-TRUE_P1*t)
        +
        np.random.normal(0,2,size=t.shape)
    )

    return (
        torch.tensor(t,dtype=torch.float32).view(-1,1),
        torch.tensor(glucose,dtype=torch.float32).view(-1,1)
    )

t_data, G_data = generate_patient_data()

# =====================================================
# PINN Model
# =====================================================

class EndocrinePINN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,2)
        )

        # Learnable physiological parameter

        self.p1 = nn.Parameter(
            torch.tensor([0.01])
        )

    def forward(self,t):

        return self.network(t)

model = EndocrinePINN()

# =====================================================
# Optimizer
# =====================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005
)

# =====================================================
# Physics Points
# =====================================================

t_phys = torch.linspace(
    0,
    120,
    200,
    requires_grad=True
).view(-1,1)

# =====================================================
# Training
# =====================================================

epochs = 4000

loss_history = []
parameter_history = []

for epoch in range(epochs):

    optimizer.zero_grad()

    # ------------------------
    # Data Loss
    # ------------------------

    pred_data = model(t_data)

    G_pred = pred_data[:,0:1]

    loss_data = nn.MSELoss()(
        G_pred,
        G_data
    )

    # ------------------------
    # Physics Loss
    # ------------------------

    pred_phys = model(t_phys)

    G_phys = pred_phys[:,0:1]
    X_phys = pred_phys[:,1:2]

    dGdt = torch.autograd.grad(
        G_phys,
        t_phys,
        grad_outputs=torch.ones_like(G_phys),
        create_graph=True
    )[0]

    residual = (
        dGdt
        +
        model.p1*(G_phys-G_B)
        +
        X_phys*G_phys
    )

    loss_phys = torch.mean(
        residual**2
    )

    # ------------------------
    # Total Loss
    # ------------------------

    loss = loss_data + 2.0*loss_phys

    loss.backward()

    optimizer.step()

    loss_history.append(
        loss.item()
    )

    parameter_history.append(
        model.p1.item()
    )

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch} | "
            f"Loss={loss.item():.4f} | "
            f"p1={model.p1.item():.5f}"
        )

# =====================================================
# Results
# =====================================================

estimated_p1 = model.p1.item()

print("\nTRUE PARAMETER")
print(TRUE_P1)

print("\nESTIMATED PARAMETER")
print(estimated_p1)

print(
    "\nRELATIVE ERROR (%)",
    abs(
        estimated_p1-TRUE_P1
    )/TRUE_P1*100
)

# =====================================================
# Trajectory Plot
# =====================================================

with torch.no_grad():

    t_test = torch.linspace(
        0,
        120,
        300
    ).view(-1,1)

    prediction = model(t_test)

    G_recon = prediction[:,0].numpy()

plt.figure(figsize=(10,6))

plt.scatter(
    t_data.numpy(),
    G_data.numpy(),
    marker='x',
    color='black',
    s=60,
    label='Clinical Observations'
)

plt.plot(
    t_test.numpy(),
    G_recon,
    linewidth=3,
    label='PINN Reconstruction'
)

plt.axhline(
    G_B,
    linestyle='--',
    color='gray',
    label='Basal Glucose'
)

plt.title(
    f'Parameter Discovery\n'
    f'True p1={TRUE_P1:.3f} '
    f'Estimated p1={estimated_p1:.3f}'
)

plt.xlabel("Time (minutes)")
plt.ylabel("Glucose (mg/dL)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "parameter_discovery.png",
    dpi=300
)

plt.show()

# =====================================================
# Parameter Convergence
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    parameter_history,
    linewidth=2
)

plt.axhline(
    TRUE_P1,
    linestyle='--'
)

plt.xlabel("Epoch")
plt.ylabel("Estimated p1")

plt.title(
    "PINN Parameter Convergence"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "loss_curve.png",
    dpi=300
)

plt.show()