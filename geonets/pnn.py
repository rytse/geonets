import torch
from torch import nn

from geonets import sympnet, inn


class PNN(nn.Module):
    def __init__(
        self,
        n,
        d,
        symp_nlayers,
        symp_subwidth,
        inn_nlayers,
        inn_nsublayers,
        inn_subwidth,
        symp_type="e",
        symp_act=nn.Sigmoid,
        inn_vp=True,
        inn_act=nn.Sigmoid,
    ):
        """
        Poisson neural network.

        Parameters:
        -----------
        n: total input dimension
        d: half the latent dimension (2d)
        symp_nlayers: number of layers in the symplectic network
        symp_subwidth: number of modules in each layer of the symplectic network
        inn_nlayers: number of layers in the invertible network
        inn_nsublayers: number of modules in each layer of the invertible network
        inn_subwidth: width of each FCN in each module of the invertible network
        symp_type: type of symplectic network ('e', 'la', or 'g')
        symp_act: activation function for symplectic network
        inn_vp: volume-preserving or non-volume preserving
        inn_act: activation function for invertible network

        """

        super().__init__()

        self.n = n
        self.d = d

        if symp_type == "e":
            self.symp = sympnet.ESympNet(
                n,
                d,
                nlayers=symp_nlayers,
                subwidth=symp_subwidth,
                act=symp_act,
            )
        elif symp_type == "la":
            if 2 * d != n:
                raise ValueError("LASympNet requires n = 2d")

            self.symp = sympnet.LASympNet(
                d,
                nlayers=symp_nlayers,
                subwidth=symp_subwidth,
                act=symp_act,
            )
        elif symp_type == "g":
            if 2 * d != n:
                raise ValueError("GSympNet requires n = 2d")

            self.symp = sympnet.GSympNet(
                d,
                nlayers=symp_nlayers,
                subwidth=symp_subwidth,
                act=symp_act,
            )
        else:
            raise ValueError("Invalid symp_type")

        self.inn = inn.INN(
            n,
            d,
            nlayers=inn_nlayers,
            nsublayers=inn_nsublayers,
            width=inn_subwidth,
            vp=inn_vp,
            act=inn_act,
        )

    def forward(self, x):
        return self.inn.inverse(self.symp(self.inn(x)))

    def moment(self, x, v):
        """
        Compute the moment map associated with this PNN at point x on the
        manifold and in the direction v in the Lie algebra associated with the
        manifold.

        Parameters:
        -----------
        x: point on the manifold
        v: direction in the Lie algebra
        """

        y = self.forward(x)
        return torch.autograd(y, x, grad_outputs=v, retain_graph=True)[0]
