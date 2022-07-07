from abc import abstractmethod
import torch
from torch import nn 
from typing import List, Union, Tuple



class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Sequential(nn.Sequential):
    def __init__(self, modules: Tuple[Module]):
        super().__init__()
        self.passed_modules = modules
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for module in self.passed_modules:
            input = module(input)
        return input


class _FCNet(Module):
    def __init__(self, indim: int, outdim: int, nlayers: int,
     width: int, act: nn.Module = nn.Sigmoid):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)




class VPINNMod_UP(Module):
    def __init__(self, n: int, d: int, nlayers: int, width: int, act=nn.Sigmoid):
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
        self.nlayers = nlayers
        self.width = width
        self.act = act

        indim = n - d
        outdim = d
        

        self.m = _FCNet(indim, outdim, nlayers, width, act)

    def forward(self, xy: torch.Tensor):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        return torch.hstack([x + self.m(y), y])
        
    def inverse(self, xy: torch.Tensor):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        return torch.hstack([x - self.m(y), y])


class VPINNMod_LOW(Module):
    def __init__(self, n: int, d: int, nlayers: int, width: int, act=nn.Sigmoid):
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
        self.nlayers = nlayers
        self.width = width
        self.act = act

        indim = d
        outdim = n - d
        

        self.m = _FCNet(indim, outdim, nlayers, width, act)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        return torch.hstack([x, y + self.m(x)])

        
    
    def inverse(self, xy: torch.Tensor):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        return torch.hstack([x, y - self.m(x)])


class NVPINNMod_UP(nn.Module):
    def __init__(self, n: int, d: int, nlayers: int, width: int, 
    act: nn.Module =nn.Sigmoid):
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
        self.nlayers = nlayers
        self.width = width
        self.act = act

        indim = n - d
        outdim = d
        

        # Construct two fully connected linear networks (self.s, self.t)
        self.s = _FCNet(indim, outdim, nlayers, width, act)
        self.t = _FCNet(indim, outdim, nlayers, width, act)

    def forward(self, xy: torch.Tensor):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        return torch.hstack([x * torch.exp(self.s(y)) + self.t(y), y])
    
    def inverse(self, xy: torch.Tensor):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        return torch.hstack([(x - self.t(y)) * torch.exp(-self.s(y)), y])


class NVPINNMod_LOW(nn.Module):
    def __init__(self, n: int, d: int, nlayers: int, width: int,
     act: nn.Module = nn.Sigmoid):
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
        self.nlayers = nlayers
        self.width = width
        self.act = act


        indim = d
        outdim = n - d
        

        # Construct two fully connected linear networks (self.s, self.t)
        self.s = _FCNet(indim, outdim, nlayers, width, act)
        self.t = _FCNet(indim, outdim, nlayers, width, act)

    def forward(self, xy: torch.Tensor):
        x = xy[:, : self.d]
        y = xy[:, self.d :]

        return torch.hstack([x, y * torch.exp(self.s(x)) + self.t(x)])

    def inverse(self, xy):
        x = xy[:, : self.d]
        y = xy[:, self.d :]


        return torch.hstack([x, (y - self.t(x)) * torch.exp(-self.s(x))])



class INN(nn.Module):
    def __init__(self, n: int, d: int, nlayers: int, nsublayers: int, 
        width: int, vp: bool = True, act: nn.Module = nn.Sigmoid):
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


        
        self.layers: List[nn.Module] = []
        ul_rep = "up"
        INN_UP_LOW = (VPINNMod_UP, VPINNMod_LOW) if vp else (NVPINNMod_UP, NVPINNMod_LOW)
        #'''
        for i in range(nlayers):
            if ul_rep == "up":
                layer = (INN_UP_LOW[0](n, d, nsublayers, width, act))     
            else:
                layer = (INN_UP_LOW[1](n, d, nsublayers, width, act)) 
            #layer = VPINNMod_LOW(n, d, nsublayers, width, act)
            
            self.layers.append(layer)
            ul_rep = "low" if ul_rep == "up" else "up"
        #'''
        #INN_UP_LOW = (VPINNMod_UP, VPINNMod_LOW)
        
        #self.layers: Tuple[VPINNMod_LOW] = tuple([VPINNMod_LOW(n,d,nsublayers, width, act)])
        
        self.M = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.M(x)

    def inverse(self, x: torch.Tensor):
        y = x
        #for i in reversed(range(self.nlayers)):
        for i in range(self.nlayers-1, -1, -1):
            y = self.layers[i].inverse(y)
        return y
