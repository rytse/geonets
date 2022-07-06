import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint

import matplotlib.pyplot as plt

from tqdm import trange

from geonets.pnn import PNN

BATCH_SIZE = 128
EPOCHS = 100_000
# EPOCHS = 1_000
LEARNING_RATE = 0.001


class EMParticleDataset(torch.utils.data.Dataset):
    def __init__(self, t_stop=60, m=1, q=1, dt=0.1):
        self.dt = dt
        self.t = torch.arange(0, t_stop, dt)
        self.T = len(self.t)

        self.m = m
        self.q = q

        self.x0_0 = torch.tensor([0.9, 0.25, -5.0, 0.5, 1.0, 40.0], requires_grad=True)
        self.x0_1 = torch.tensor([0.9, 0.25, 0.7, 0.5, 1.0, 10.0], requires_grad=True)
        self.x0_2 = torch.tensor([1.1, 0.3, -0.7, 0.5, 1.0, 20.0], requires_grad=True)

        # Integrate the physics
        self.z_0 = odeint(self.step, self.x0_0, self.t)
        self.z_0 = self.z_0.detach()
        self.z_1 = odeint(self.step, self.x0_1, self.t)
        self.z_1 = self.z_1.detach()
        self.z_2 = odeint(self.step, self.x0_2, self.t)
        self.z_2 = self.z_2.detach()

        # Concatenate the data
        self.z = torch.stack([self.z_0, self.z_1, self.z_2], dim=0)

    def H(self, z):
        v = z[0:3]
        x = z[3:6]

        return 0.5 * self.m * torch.inner(v, v) + self.q / 100.0 / torch.norm(x)

    def step(self, t, z):
        y = self.H(z)
        DH = torch.autograd.grad(y, z, retain_graph=True)[0]

        v = z[0:3]
        x = z[3:6]
        B_hat = torch.tensor([[0, -torch.norm(x), 0], [torch.norm(x), 0, 0], [0, 0, 0]])
        I_hat = torch.eye(3)

        top = torch.cat([-self.q / self.m ** 2 * B_hat, -1 / self.m * I_hat], dim=0)
        bottom = torch.cat([1 / self.m * I_hat, torch.zeros(3, 3)], dim=0)
        B = torch.cat([top, bottom], dim=1)

        return self.dt * B @ DH

    def __len__(self):
        return (self.T - 1) * 3

    def __getitem__(self, idx):
        traj_idx = idx // (self.T - 1)

        start = self.z[traj_idx, idx % (self.T - 1), :]
        end = self.z[traj_idx, (idx % (self.T - 1)) + 1, :]

        return start, end


pd = EMParticleDataset()
znp = pd.z.detach().numpy()
trainloader = torch.utils.data.DataLoader(pd, batch_size=BATCH_SIZE, shuffle=True)

try:
    pnn = torch.load("./geonets/test/output/em_particle.pt")
    print("Loaded existing model")
except FileNotFoundError:
    pnn = PNN(
        6,
        2,
        symp_nlayers=3,
        symp_subwidth=4,
        inn_nlayers=3,
        inn_nsublayers=3,
        inn_subwidth=4,
        symp_type="e",
    )
    print("Created new model")

criterion = nn.MSELoss()
optimizer = optim.Adam(pnn.parameters(), lr=LEARNING_RATE)

with trange(EPOCHS) as tepoch:
    for epoch in tepoch:
        for i, data in enumerate(trainloader, 0):
            inputs, outputs = data

            optimizer.zero_grad()
            pred_outputs = pnn(inputs)

            loss = criterion(pred_outputs, outputs)
            loss.backward(retain_graph=True)
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

torch.save(pnn, "./geonets/test/output/em_particle.pt")

N_NEXT = 1000
x_next = torch.zeros(N_NEXT, 6)
x_cur = torch.tensor([znp[1, -1, :]])

for i in range(N_NEXT):
    x_cur = pnn(x_cur)
    x_next[i, :] = x_cur[0]

x_next = x_next.detach().numpy()

ax = plt.axes(projection="3d")
# ax.plot3D(znp[0, :, 3], znp[0, :, 4], znp[0, :, 5], "b")
ax.plot3D(znp[1, :, 3], znp[1, :, 4], znp[1, :, 5], "b")
# ax.plot3D(znp[2, :, 3], znp[2, :, 4], znp[2, :, 5], "b")
plt.plot(x_next[:, 0], x_next[:, 1], "r")
plt.title("Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
