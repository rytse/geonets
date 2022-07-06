import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint

import matplotlib.pyplot as plt

from tqdm import trange

from geonets.pnn import PNN

BATCH_SIZE = 128
EPOCHS = 250_000
# EPOCHS = 1_000
LEARNING_RATE = 0.001


class LotkaVolterraDataset(torch.utils.data.Dataset):
    def __init__(self, t_stop=60, dt=0.1):
        self.dt = dt
        self.t = torch.arange(0, t_stop, dt)
        self.T = len(self.t)

        self.x0_0 = torch.tensor([1.0, 0.8], requires_grad=True)
        self.x0_1 = torch.tensor([1.0, 1.0], requires_grad=True)
        self.x0_2 = torch.tensor([1.0, 1.2], requires_grad=True)

        # Integrate the physics
        self.z_0 = odeint(self.step, self.x0_0, self.t)
        self.z_0 = self.z_0.detach()
        self.z_1 = odeint(self.step, self.x0_1, self.t)
        self.z_1 = self.z_1.detach()
        self.z_2 = odeint(self.step, self.x0_2, self.t)
        self.z_2 = self.z_2.detach()

        # Concatenate the data
        self.z = torch.stack([self.z_0, self.z_1, self.z_2], dim=0)

    def H(self, x):
        return x[0] - torch.log(x[0]) + x[1] - 2 * torch.log(x[1])

    def step(self, t, x):
        y = self.H(x)
        DH = torch.autograd.grad(y, x, retain_graph=True)[0]
        B = torch.tensor([[0, x[0] * x[1]], [-x[0] * x[1], 0]])
        return self.dt * B @ DH

    def __len__(self):
        return (self.T - 1) * 3

    def __getitem__(self, idx):
        traj_idx = idx // (self.T - 1)

        start = self.z[traj_idx, idx % (self.T - 1), :]
        end = self.z[traj_idx, (idx % (self.T - 1)) + 1, :]

        return start, end


pd = LotkaVolterraDataset()
znp = pd.z.detach().numpy()
trainloader = torch.utils.data.DataLoader(pd, batch_size=BATCH_SIZE, shuffle=True)

try:
    pnn = torch.load("./geonets/test/output/lotka_volterra.pt")
    print("Loaded existing model")
except FileNotFoundError:
    pnn = PNN(
        2,
        1,
        symp_nlayers=6,
        symp_subwidth=6,
        inn_nlayers=4,
        inn_nsublayers=4,
        inn_subwidth=6,
        symp_type="la",
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

torch.save(pnn, "./geonets/test/output/lotka_volterra.pt")

N_NEXT = 1000
x_next = torch.zeros(N_NEXT, 2)
x_cur = torch.tensor([znp[0, -1, :]])

for i in range(N_NEXT):
    x_cur = pnn(x_cur)
    x_next[i, :] = x_cur[0]

x_next = x_next.detach().numpy()


plt.plot(znp[0, :, 0], znp[0, :, 1], "b")
plt.plot(znp[1, :, 0], znp[1, :, 1], "b")
plt.plot(znp[2, :, 0], znp[2, :, 1], "b")
plt.plot(x_next[:, 0], x_next[:, 1], "r")
plt.title("Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
