from abc import abstractmethod

import torch
from torch import nn
from typing import List, Union


class LinSympLayer_UP(nn.Module):
    def __init__(self, d: int):
        """
        Layer in a linear module for a symplectic network. Parameterizes a small
        subset of linear mappings that are valid symplectic forms. These layers
        are composed into modules which are composed with other "symplectic
        modules" to form full symplectic networks.

        Parameters:
        -----------
        d: dimension of the linear layer
        ul: 'up' or 'low', upper or lower module

        """

        super().__init__()

        self.d = d

        self.A = nn.Parameter(torch.randn(d, d))

        self.M = torch.eye(2 * self.d, 2 * self.d)
        
    def forward(self, pq: torch.Tensor):
        S = self.A + self.A.T

        self.M[self.d:, :self.d] = S.detach()
        

        return self.M @ pq


class LinSympLayer_LOW(nn.Module):
    def __init__(self, d: int):
        """
        Layer in a linear module for a symplectic network. Parameterizes a small
        subset of linear mappings that are valid symplectic forms. These layers
        are composed into modules which are composed with other "symplectic
        modules" to form full symplectic networks.

        Parameters:
        -----------
        d: dimension of the linear layer
        ul: 'up' or 'low', upper or lower module

        """

        super().__init__()

        self.d = d

        self.A = nn.Parameter(torch.randn(d, d))

        self.M = torch.eye(2 * self.d, 2 * self.d)
        
    def forward(self, pq: torch.Tensor):
        S = self.A + self.A.T

        self.M[:self.d, self.d:] = S.detach()
        

        return self.M @ pq



def LinSympLayer(d: int, ul: str):
    if ul == "up":
        return (LinSympLayer_UP(d))
    
    return (LinSympLayer_LOW(d))




class LinSympMod(nn.Module):
    def __init__(self, d: int, width: int, ul_rep: str):
        """
        Linear module for symplectic networks (SympNets). Parameterizes a set
        of linear mappings that are valid symplectic forms. These modules are
        composed with other "symplectic modules" to form full sympletic
        networks.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        width: module width, i.e. number of transformations to compose
        ul: 'up' or 'low', upper or lower module (which to multiply by first)

        """

        super().__init__()

        self.d = d
        self.width = width

        self.b = nn.Parameter(torch.randn(2 * d))

        # Compose linear symplectic layers. Note that this isn't exactly
        # the same notion as a layer in a regular neural network, this is
        # a "wide" sequence of multiplications.
        self.layers: List[Union[LinSympLayer_LOW, LinSympLayer_UP]] = []
        for i in range(width):
            self.layers.append(LinSympLayer(d, ul_rep))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.layers)

    def forward(self, pq: torch.Tensor):
        sequential_forward: torch.Tensor = self.M(pq.T)
        return sequential_forward.T + self.b

class ActSympMod_UP(nn.Module):
    def __init__(self, d: int, act: nn.Module):
        """
        Activation module for symplectic networks (SympNets). Parameterizes the
        set of activation functions that preserve symplecticity when applied to
        linear symplectic modules. These modules are composed with other
        "symplectic modules" to form full sympletic networks.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        ul: 'up' or 'low', upper or lower module
        act: activation function

        """
        super().__init__()

        self.d = d

        self.act = act

        self.a = nn.Parameter(torch.randn(d))


    def forward(self, pq: torch.Tensor):
        p = pq[:, : self.d]
        q = pq[:, self.d :]

        return torch.hstack([p + self.a * self.act(q), q])

class ActSympMod_LOW(nn.Module):
    def __init__(self, d: int, act: nn.Module):
        """
        Activation module for symplectic networks (SympNets). Parameterizes the
        set of activation functions that preserve symplecticity when applied to
        linear symplectic modules. These modules are composed with other
        "symplectic modules" to form full sympletic networks.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        ul: 'up' or 'low', upper or lower module
        act: activation function

        """
        super().__init__()

        self.d = d

        self.act = act

        self.a = nn.Parameter(torch.randn(d))


    def forward(self, pq: torch.Tensor):
        p = pq[:, :self.d]
        q = pq[:, self.d:]

        return torch.hstack([p, q + self.a * self.act(p)])


def ActSympMod(d: int, ul: str, act: nn.Module):
    if ul == "up":
        return (ActSympMod_UP(d, act))
    return (ActSympMod_LOW(d, act))



class GradSympMod_UP(nn.Module):
    def __init__(self, d: int, width: int, act: nn.Module):
        """
        Gradient module for symplectic networks (SympNets). Parameterizes a
        subset of quadratic forms that are valid symplectic forms. This is a
        potentially faster alternative to linear modules. These modules are
        composed with other "symplectic modules" to form full sympletic
        networks.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        width: module width, number of multiplications before summing
        ul: 'up' or 'low', upper or lower module
        act: activation function

        """

        super().__init__()

        self.d = d
        self.width = width

        self.act = act

        self.K = nn.Parameter(torch.randn(width, d))
        self.a = nn.Parameter(torch.randn(width))
        self.b = nn.Parameter(torch.randn(width))


    def forward(self, pq: torch.Tensor):
        P = pq[:, : self.d]
        Q = pq[:, self.d :]

     

        x: torch.Tensor = torch.einsum("ij, kj -> ki", self.K, Q)
        y: torch.Tensor = self.a * self.act(x + self.b)
        z = (self.K.T @ y.T ).T

        return torch.hstack([P + z, Q])
        

class GradSympMod_LOW(nn.Module):
    def __init__(self, d: int, width: int, act: nn.Module):
        """
        Gradient module for symplectic networks (SympNets). Parameterizes a
        subset of quadratic forms that are valid symplectic forms. This is a
        potentially faster alternative to linear modules. These modules are
        composed with other "symplectic modules" to form full sympletic
        networks.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        width: module width, number of multiplications before summing
        ul: 'up' or 'low', upper or lower module
        act: activation function

        """

        super().__init__()

        self.d = d
        self.width = width

        self.act = act

        self.K = nn.Parameter(torch.randn(width, d))
        self.a = nn.Parameter(torch.randn(width))
        self.b = nn.Parameter(torch.randn(width))


    def forward(self, pq: torch.Tensor):
        P = pq[:, : self.d]
        Q = pq[:, self.d :]

        x: torch.Tensor = torch.einsum("ij, kj -> ki", self.K, P)
        y: torch.Tensor = self.a * self.act(x + self.b)
        z = (self.K.T @ y.T).T

        return torch.hstack([P, Q + z])


def GradSympMod(d: int, width: int, ul: str, act: nn.Module):
    if ul == "up":
        return GradSympMod_UP(d, width, act)
    
    return GradSympMod_LOW(d, width, act)



class ExtSympMod_UP(nn.Module):
    def __init__(self, n: int, d: int, width: int, act: nn.Module):
        """
        Extended module for symplectic networks (SympNets). Parameterizes a set
        of quadratic + affine forms that are valid symplectic forms. These
        modules are composed with other "symplectic modules" to form full
        sympletic networks.

        Parameters:
        -----------
        n: total input dimension
        d: reduction dimension
        width: module width, number of multiplications before summing
        ul: 'up' or 'low', upper or lower module
        act: activation function

        """

        super().__init__()

        self.n = n
        self.d = d
        self.width = width
        self.act = act

        self.K1 = nn.Parameter(torch.randn(width, d))
        self.K2 = nn.Parameter(torch.randn(width, n - 2 * d))
        self.a = nn.Parameter(torch.randn(width))
        self.b = nn.Parameter(torch.randn(width))

    

    def forward(self, pqc: torch.Tensor):
        P = pqc[:, : self.d]
        Q = pqc[:, self.d : 2 * self.d]
        C = pqc[:, 2 * self.d :]

        x = self.K1 @ Q.T + self.K2 @ C.T + self.b[:, None]
        y: torch.Tensor = self.a[:, None] * self.act(x)
        z = (self.K1.T @ y).T
        return torch.hstack([P + z, Q, C])


class ExtSympMod_LOW(nn.Module):
    def __init__(self, n: int, d: int, width: int, act: nn.Module):
        """
        Extended module for symplectic networks (SympNets). Parameterizes a set
        of quadratic + affine forms that are valid symplectic forms. These
        modules are composed with other "symplectic modules" to form full
        sympletic networks.

        Parameters:
        -----------
        n: total input dimension
        d: reduction dimension
        width: module width, number of multiplications before summing
        ul: 'up' or 'low', upper or lower module
        act: activation function

        """

        super().__init__()

        self.n = n
        self.d = d
        self.width = width
        self.act = act

        self.K1 = nn.Parameter(torch.randn(width, d))
        self.K2 = nn.Parameter(torch.randn(width, n - 2 * d))
        self.a = nn.Parameter(torch.randn(width))
        self.b = nn.Parameter(torch.randn(width))

    

    def forward(self, pqc: torch.Tensor):
        P = pqc[:, : self.d]
        Q = pqc[:, self.d : 2 * self.d]
        C = pqc[:, 2 * self.d :]

        x = self.K1 @ P.T + self.K2 @ C.T + self.b[:, None]
        y: torch.Tensor = self.a[:, None] * self.act(x)
        z = (self.K1.T @ y).T
        return torch.hstack([P, Q + z, C])

def ExtSympMod(n: int, d: int, width: int, ul: str, act: nn.Module):
    if ul == "up":
        return (ExtSympMod_UP(n, d, width, act))
    
    return (ExtSympMod_LOW(n, d, width, act))


class SympNet(nn.Module):
    """
    Abstract class for symplectic networks (SympNets).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass



class LASympNet(SympNet):
    def __init__(self, d: int, nlayers: int, subwidth: int, act: nn.Module = nn.Sigmoid):
        """
        Linear + activation module based symplectic network. Made of a set of
        linear symplectic modules and a set of activation modules.

        TODO determine why in the paper they don't use an activation on the
        last layer.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        nlayers: number of layers of act(lin()) modules
        subwidth: width of each linear module
        act: activation function (pass a constructor)

        """

        super().__init__()

        self.d = d
        self.nlayers = nlayers
        self.subwidth = subwidth

        ul_rep = "up"
        self.layers = []
        for i in range(nlayers):
            self.layers.append(LinSympMod(d, subwidth, ul_rep))
            self.layers.append(ActSympMod(d, ul_rep, act()))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.layers)

    def forward(self, pq):
        return self.M(pq)


class GSympNet(SympNet):
    def __init__(self, d: int, nlayers: int, subwidth: int, act: nn.Module =nn.Sigmoid):
        """
        Gradient module based symplectic network. Made of a set of gradient
        symplectic modules.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        nlayers: number of layers of act(lin()) modules
        subwidth: width of each linear module
        act: activation function (pass a constructor)
        """

        super().__init__()

        self.d = d
        self.nlayers = nlayers
        self.subwidth = subwidth

        ul_rep = "up"
        self.layers = []
        for i in range(nlayers):
            self.layers.append(GradSympMod(d, subwidth, ul_rep, act()))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.layers)

    def forward(self, pq):
        return self.M(pq)



class ESympNet(SympNet):
    def __init__(self, n: int, d: int, nlayers: int, subwidth: int, act: nn.Module = nn.Sigmoid):
        """
        Extended module based symplectic network. Made of a set of extended
        symplectic modules.

        Parameters:
        -----------
        n: total input dimension
        d: reduction dimension
        nlayers: number of layers of act(lin()) modules
        subwidth: width of each linear module
        act: activation function (pass a constructor)
        """

        super().__init__()

        self.n = n
        self.d = d
        self.nlayers = nlayers
        self.subwidth = subwidth

        ul_rep = "up"
        self.layers = []
        for i in range(nlayers):
            self.layers.append(ExtSympMod(n, d, subwidth, ul_rep, act()))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.layers)

    def forward(self, pqc):
        return self.M(pqc)