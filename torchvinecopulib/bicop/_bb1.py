import torch
from ._archimedean import BiCopArchimedean

def _g(vec: torch.Tensor, delta: float) -> torch.Tensor:
    return (-vec * delta).expm1()

class Bb1(BiCopArchimedean):

    _PAR_MIN, _PAR_MAX = (0, 0.99), (6.99, 6.99)

    @staticmethod
    # Joe 2014 page 190
    # compare with pow function
    def cdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta = par[0]
        delta = par[1]
        return (
                    (-1/theta)*(
                        ((1/delta)*(
                            (delta * _g(obs[:, [0]].log(), theta).log()).exp() 
                            + (delta * _g(obs[:, [1]].log(), theta).log()).exp()
                            ).log()
                        ).exp()
                    ).log1p()
                ).exp()
    

    @staticmethod
    def generator(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta = par[0]
        delta = par[1]
        return (
            delta * (
                    _g(obs[:, [0]].log(), theta)
                ).log()
        ).exp()
        
    @staticmethod
    def generator_inv(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta = par[0]
        delta = par[1]
        return (
            (-1/theta) * (
                ((1/delta) * obs[:, [0]].log()).exp()
            ).log1p()
        ).exp()
    
    
    @staticmethod
    def generator_derivative(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta = par[0]
        delta = par[1]
        return (
            -delta * theta * (
                obs[:, [0]].log() * (-1 - theta)
            ).exp() * (
                _g(obs[:, [0]].log(), -theta).log() * (delta - 1)
            ).exp()
        )
    
    
    @staticmethod
    def pars2tau(par: tuple[float]) -> torch.Tensor:
        return 1 - 2 / (par(1) * (par(0) + 2))
    
    

