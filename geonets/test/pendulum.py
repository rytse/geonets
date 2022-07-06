import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint

import matplotlib.pyplot as plt

from tqdm import trange

import geonets.sympnet as sn

BATCH_SIZE = 128
EPOCHS = 500_000
LEARNING_RATE = 0.001


class PendulumDataset(torch.utils.data.Dataset):
    def __init__(self, t_stop=20, dt=0.1):
        self.dt = dt
        self.t = torch.arange(0, t_stop, dt)
        self.x0 = torch.tensor([0.0, 1.0], requires_grad=True)

        # Set up symplectic integrator
        d = 1
        self.J = torch.zeros(2 * d, 2 * d)
        self.J[d:, :d] = -1 * torch.eye(d)
        self.J[:d, d:] = torch.eye(d)

        # Integrate the physics
        self.z = odeint(self.step, self.x0, self.t)
        self.z = self.z.detach()

    def H(self, x):
        return 0.5 * x[0] * x[0] - torch.cos(x[1])

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


pd = PendulumDataset()
znp = pd.z.detach().numpy()
trainloader = torch.utils.data.DataLoader(pd, batch_size=BATCH_SIZE, shuffle=True)

try:
    sympnet = torch.load("./geonets/test/output/pendulum_sympnet.pt")
    print("Loaded existing model")
except FileNotFoundError:
    sympnet = sn.LASympNet(1, nlayers=3, subwidth=4)
    # sympnet = sn.GSympNet(1, nlayers=3, subwidth=4)
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

torch.save(sympnet, "./geonets/test/output/pendulum_sympnet.pt")

N_NEXT = 1000
x_next = torch.zeros(N_NEXT, 2)
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
