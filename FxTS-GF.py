"""
@author: mayank baranwal
"""
import torch
from torch.optim import Optimizer

class FxTS_Momentum(Optimizer):
    """ Implements FxTS-GF optimizer with momentum
    
    Parameters:
        lr (float): learning rate. Default 1e-3
        betas (tuple of two floats): FxTS beta parameters (b1,b2). Default: (0.9,0.9)
        alphas (tuple of two floats): FxTS alpha parameters (a1,a2). Default: (20,1.98)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9,0.9), alphas=(20,1.98), momentum=0.3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if betas[0] < 0.0:
            raise ValueError("Invalid beta parameter: {} - should be >= 0.0".format(betas[0]))
        if betas[1] < 0.0:
            raise ValueError("Invalid beta parameter: {} - should be >= 0.0".format(betas[1]))
        if not alphas[0] > 2.0:
            raise ValueError("Invalid alpha parameter: {} - should be > 2.0".format(alphas[0]))
        if not 1.0 < alphas[1] < 2.0:
            raise ValueError("Invalid alpha parameter: {} - should be >= in [1.01, 1.99]".format(alphas[1]))
        if not 0.0 < momentum < 0.5:
            raise ValueError("Invalid momentum parameter: {} - should be >= in [0, 0.5]".format(momentum))
            
        defaults = dict(lr=lr, betas=betas, alphas=alphas, momentum=momentum)
        super(FxTS_Momentum, self).__init__(params, defaults)
        
    
    def __setstate__(self, state):
        super(FxTS_Momentum, self).__setstate__(state)
        
    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
            
        for group in self.param_groups:
            beta1, beta2   = group['betas']
            alpha1, alpha2 = group['alphas']
            lr             = group['lr']
            momentum       = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad      = p.grad.data
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    
                    state['v'] = torch.zeros_like(p.data)
                    
                v = state['v']
                state['step'] += 1
                v.mul_(momentum).add_(1-momentum,grad)
                
                v_norm = v.norm()
                factor = beta1 * (v_norm ** (1 - (alpha1-2)/(alpha1-1))) + beta2 * (v_norm ** (1 - (alpha2-2)/(alpha2-1)))
                torch.sign(v).mul_(factor)
                
                p.data.add_(-lr,v)
                
        return loss
