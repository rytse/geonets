import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint

import matplotlib.pyplot as plt

from tqdm import trange

import geonets.sympnet as sn

BATCH_SIZE = 512
EPOCHS = 1_000_000
LEARNING_RATE = 0.001


class DoublePendulumDataset(torch.utils.data.Dataset):
    def __init__(self, t_stop=50, dt=0.1, m1=1, m2=1, l1=1, l2=1, g=9.8):
        self.dt = dt
        self.t = torch.arange(0, t_stop, dt)
        self.x0 = torch.tensor([0.0, 1.0, 0.0, 1.0], requires_grad=True)

        # Simulation parameters
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

        # Set up symplectic integrator
        d = 2
        self.J = torch.zeros(2 * d, 2 * d)
        self.J[d:, :d] = -1 * torch.eye(d)
        self.J[:d, d:] = torch.eye(d)

        # Integrate the physics
        self.z = odeint(self.step, self.x0, self.t)
        self.z = self.z.detach()

    def H(self, x):
        z1 = self.m2 * pow(self.l2, 2) * pow(x[0], 2)
        z2 = (self.m1 + self.m2) * pow(self.l1, 2) * pow(x[1], 2)
        z3 = -2.0 * self.m2 * self.l1 * self.l2 * x[0] * x[1] * torch.cos(x[2] - x[3])
        z4 = 2.0 * self.m2 * pow(self.l1, 2) * pow(self.l2, 2)
        z5 = self.m1 + self.m2 * pow(torch.sin(x[2] - x[3]), 2)
        z6 = -1.0 * (self.m1 + self.m2) * self.g * self.l1 * torch.cos(x[2])
        z7 = -1.0 * self.m2 * self.g * self.l2 * torch.cos(x[3])

        return (z1 + z2 + z3) / z4 / z5 + z6 + z7

    def step(self, t, x):
        y = self.H(x)
        DH = torch.autograd.grad(y, x, retain_graph=True)[0]
        return self.dt * -1 * self.J @ DH

    def __len__(self):
        return len(self.t) - 1

    def __getitem__(self, idx):
        start = self.z[idx, :]
        end = self.z[idx + 1, :]
        return start, end


pd = DoublePendulumDataset()
znp = pd.z.detach().numpy()
trainloader = torch.utils.data.DataLoader(pd, batch_size=BATCH_SIZE, shuffle=True)

try:
    sympnet = torch.load("./geonets/test/output/double_pendulum_sympnet.pt")
    print("Loaded existing model")
except FileNotFoundError:
    sympnet = sn.LASympNet(2, nlayers=3, subwidth=4)
    # sympnet = sn.GSympNet(2, nlayers=3, subwidth=4)
    # sympnet = sn.ESympNet(4, 1, nlayers=3, subwidth=4)
    print("Created new model")

criterion = nn.MSELoss()
optimizer = optim.Adam(sympnet.parameters(), lr=LEARNING_RATE)

with trange(EPOCHS) as tepoch:
    for epoch in tepoch:
        for i, data in enumerate(trainloader, 0):
            inputs, outputs = data

            optimizer.zero_grad()
            pred_outputs = sympnet(inputs)

            loss = criterion(pred_outputs, outputs)
            loss.backward(retain_graph=True)
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

torch.save(sympnet, "./geonets/test/output/double_pendulum_sympnet.pt")

N_NEXT = 1000
x_next = torch.zeros(N_NEXT, 4)
x_cur = torch.tensor([znp[-1, :]])

for i in range(N_NEXT):
    x_cur = sympnet(x_cur)
    x_next[i, :] = x_cur[0]

x_next = x_next.detach().numpy()

plt.plot(znp[:, 0], znp[:, 1], "b")
plt.plot(x_next[:, 0], x_next[:, 1], "r")
plt.title("Pendulum Trajectory: Ground Truth")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
