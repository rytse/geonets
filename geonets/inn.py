from abc import abstractmethod
import torch
from torch import nn


class FCNet(nn.Module):
    def __init__(self, indim, outdim, nlayers, width, act=nn.Sigmoid):
        """
        Fully connected neural network with hidden layers.

        Parameters:
        ----------
        indim: input dimension
        outdim: output dimension
        nlayers: number of hidden layers
        width: dimension of each hidden layer
        act: activation function

        """

        super().__init__()

        self.layers = []
        self.layers.append(nn.Linear(indim, width))
        self.layers.append(act())
        for i in range(nlayers):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(act())
        self.layers.append(nn.Linear(width, outdim))
        self.layers.append(act())

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


class VPINNMod(nn.Module):
    def __init__(self, n, d, ul, nlayers, width, act=nn.Sigmoid):
        """
        Volume-preserving invertible neural net module. Individual modules are
        put together to assemble a full invertible neural net (INN).

        Parameters:
        -----------
        n: total dim of the INN (input and output)
        d: reduction dim (corresponds to 2d latent dim)
        ul: 'up' or 'low', upper or lower module
        nlayers: number of layers in the module
        width: module width
        act: activation function

        """

        super().__init__()

        self.n = n
        self.d = d
        self.ul = ul
        self.nlayers = nlayers
        self.width = width
        self.act = act

        if self.ul == "up":
            indim = n - d
            outdim = d
        elif self.ul == "low":
            indim = d
            outdim = n - d
        else:
            raise ValueError("ul must be 'up' or 'low'")

        self.m = FCNet(indim, outdim, nlayers, width, act)

    def forward(self, xy):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        if self.ul == "up":
            return torch.hstack([x + self.m(y), y])
        elif self.ul == "low":
            return torch.hstack([x, y + self.m(x)])

    def inverse(self, xy):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        if self.ul == "up":
            return torch.hstack([x - self.m(y), y])
        elif self.ul == "low":
            return torch.hstack([x, y - self.m(x)])


class NVPINNMod(nn.Module):
    def __init__(self, n, d, ul, nlayers, width, act=nn.Sigmoid):
        """
        Non-volume preserving invertible neural net module. Individual modules
        are put together to assemble a full invertible neural net (INN).

        Parameters:
        -----------
        n: total dim of the INN (input and output)
        d: reduction dim (corresponds to 2d latent dim)
        ul: 'up' or 'low', upper or lower module
        nlayers: number of layers in the module
        width: module width
        act: activation function

        """

        super().__init__()

        self.n = n
        self.d = d
        self.ul = ul
        self.nlayers = nlayers
        self.width = width
        self.act = act

        if self.ul == "up":
            indim = n - d
            outdim = d
        elif self.ul == "low":
            indim = d
            outdim = n - d
        else:
            raise ValueError("ul must be 'up' or 'low'")

        # Construct two fully connected linear networks (self.s, self.t)
        self.s = FCNet(indim, outdim, nlayers, width, act)
        self.t = FCNet(indim, outdim, nlayers, width, act)

    def forward(self, xy):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        if self.ul == "up":
            return torch.hstack([x * torch.exp(self.s(y)) + self.t(y), y])
        elif self.ul == "low":
            return torch.hstack([x, y * torch.exp(self.s(x)) + self.t(x)])

    def inverse(self, xy):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        if self.ul == "up":
            return torch.hstack([(x - self.t(y)) * torch.exp(-self.s(y)), y])
        elif self.ul == "low":
            return torch.hstack([x, (y - self.t(x)) * torch.exp(-self.s(x))])


class INN(nn.Module):
    def __init__(self, n, d, nlayers, nsublayers, width, vp=True, act=nn.Sigmoid):
        """
        Invertible neural net.

        Parameters:
        -----------
        n: total dim of the INN (input and output)
        d: reduction dim (corresponds to 2d latent dim)
        nlayers: number of layers in the module
        nsublayers: number of sublayers in each module
        width: module width
        vp: volume-preserving or non-volume preserving
        act: activation function (pass a constructor)

        """

        super().__init__()

        self.n = n
        self.d = d
        self.nlayers = nlayers
        self.nsublayers = nsublayers
        self.width = width
        self.act = act

        if vp:
            inn_mod = VPINNMod
        else:
            inn_mod = NVPINNMod

        self.modules = []
        ul_rep = "up"
        for i in range(nlayers):
            self.modules.append(inn_mod(n, d, ul_rep, nsublayers, width, act))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.modules)

    def forward(self, x):
        return self.M(x)

    def inverse(self, x):
        y = x
        for i in reversed(range(self.nlayers)):
            y = self.modules[i].inverse(y)
        return y
