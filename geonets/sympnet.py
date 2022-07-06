from abc import abstractmethod

import torch
from torch import nn


class LinSympLayer(nn.Module):
    def __init__(self, d, ul):
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
        self.ul = ul

        self.A = nn.Parameter(torch.randn(d, d))

        if ul != "up" and ul != "low":
            raise ValueError("ul must be 'up' or 'low'")

    def forward(self, pq):
        S = self.A + self.A.t()
        M = torch.eye(2 * self.d, 2 * self.d)

        if self.ul == "up":
            M[self.d :, : self.d] = S
        elif self.ul == "low":
            M[: self.d, self.d :] = S

        return M @ pq


class LinSympMod(nn.Module):
    def __init__(self, d, width, ul):
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
        self.ul = ul
        self.width = width

        self.b = nn.Parameter(torch.randn(2 * d))

        # Compose linear symplectic layers. Note that this isn't exactly
        # the same notion as a layer in a regular neural network, this is
        # a "wide" sequence of multiplications.
        self.layers = []
        ul_rep = ul
        for i in range(width):
            self.layers.append(LinSympLayer(d, ul_rep))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.layers)

    def forward(self, pq):
        return self.M(pq.t()).t() + self.b


class ActSympMod(nn.Module):
    def __init__(self, d, ul, act):
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

        self.ul = ul
        self.act = act

        self.a = nn.Parameter(torch.randn(d))

        if ul != "up" and ul != "low":
            raise ValueError("ul must be 'up' or 'low'")

    def forward(self, pq):
        p = pq[:, : self.d]
        q = pq[:, self.d :]

        if self.ul == "up":
            return torch.hstack([p + self.a * self.act(q), q])
        elif self.ul == "low":
            return torch.hstack([p, q + self.a * self.act(p)])


class GradSympMod(nn.Module):
    def __init__(self, d, width, ul, act):
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

        self.ul = ul
        self.act = act

        self.K = nn.Parameter(torch.randn(width, d))
        self.a = nn.Parameter(torch.randn(width))
        self.b = nn.Parameter(torch.randn(width))

        if self.ul != "up" and self.ul != "low":
            raise ValueError("ul must be 'up' or 'low'")

    def forward(self, pq):
        P = pq[:, : self.d]
        Q = pq[:, self.d :]

        if self.ul == "up":
            # z = self.K.t() @ (self.a * self.act(self.K @ q.t() + self.b))
            # but batched

            x = torch.einsum("ij, kj -> ki", self.K, Q)
            y = self.a * self.act(x + self.b)
            z = (self.K.t() @ y.t()).t()

            return torch.hstack([P + z, Q])
        elif self.ul == "low":
            # z = self.K.t() @ (self.a * self.act(self.K @ p + self.b))
            # but batched

            x = torch.einsum("ij, kj -> ki", self.K, P)
            y = self.a * self.act(x + self.b)
            z = (self.K.t() @ y.t()).t()

            return torch.hstack([P, Q + z])


class ExtSympMod(nn.Module):
    def __init__(self, n, d, width, ul, act):
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
        self.ul = ul
        self.act = act

        self.K1 = nn.Parameter(torch.randn(width, d))
        self.K2 = nn.Parameter(torch.randn(width, n - 2 * d))
        self.a = nn.Parameter(torch.randn(width))
        self.b = nn.Parameter(torch.randn(width))

        if self.ul != "up" and self.ul != "low":
            raise ValueError("ul must be 'up' or 'low'")

    def forward(self, pqc):
        P = pqc[:, : self.d]
        Q = pqc[:, self.d : 2 * self.d]
        C = pqc[:, 2 * self.d :]

        if self.ul == "up":
            x = self.K1 @ Q.t() + self.K2 @ C.t() + self.b[:, None]
            y = self.a[:, None] * self.act(x)
            z = (self.K1.t() @ y).t()
            return torch.hstack([P + z, Q, C])
        elif self.ul == "low":
            x = self.K1 @ P.t() + self.K2 @ C.t() + self.b[:, None]
            y = self.a[:, None] * self.act(x)
            z = (self.K1.t() @ y).t()
            return torch.hstack([P, Q + z, C])


class SympNet(nn.Module):
    """
    Abstract class for symplectic networks (SympNets).
    """

    @abstractmethod
    def forward(self, x):
        pass


class LASympNet(SympNet):
    def __init__(self, d, nlayers, subwidth, act=nn.Sigmoid):
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
    def __init__(self, d, nlayers, subwidth, act=nn.Sigmoid):
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
    def __init__(self, n, d, nlayers, subwidth, act=nn.Sigmoid):
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
