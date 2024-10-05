import torch
from ._archimedean import BiCopArchimedean

def _g(vec: torch.Tensor, delta: float) -> torch.Tensor:
    return (-vec * delta).expm1()

class Bb1(BiCopArchimedean):

    _PAR_MIN, _PAR_MAX = (0, 1.01), (6.99, 6.99)
    

    @staticmethod
    def generator(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return pow(pow(obs[:,[0]], par[0]) -1, par[1])
        
    @staticmethod
    def generator_inv(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return pow(pow(obs[:,[0]], 1 / par[1]) + 1, -1 / par[0])
    
    
    @staticmethod
    def generator_derivative(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta = par[0]
        delta = par[1]
        res = -delta * theta * pow(obs[:,[0]], -(1 + theta))
        res *= pow(pow(obs[:,[0]], -theta) - 1, delta - 1)
        return res
    
    @staticmethod
    def pars2tau(par: tuple[float]) -> torch.Tensor:
        return 1 - 2 / (par(1) * (par(0) + 2))
    
    

