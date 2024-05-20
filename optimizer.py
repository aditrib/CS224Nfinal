from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                ### 

                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                m_prev = state.get('m', torch.zeros_like(p.data, device=DEVICE))  # default to initializations of m,v,t for 1st iter
                v_prev = state.get('v', torch.zeros_like(p.data, device=DEVICE))
                grad = grad.to(DEVICE)
                t = state.get('t', 0)
                
                t = t + 1
                mt = beta1 * m_prev + (1 - beta1) * grad
                vt = beta2 * v_prev + (1 - beta2) * grad ** 2

                alphat = alpha * math.sqrt(1 - beta2**t) / (1 - beta1 **t)
                p.data.add_(-alphat * mt / (vt.sqrt() + eps))

                # weight decay, update loss and gradient
                # loss += 0.5 * weight_decay * torch.sum(p.data * p.data)
                p.data -= alpha * weight_decay * p.data

                # update state
                state['m'] = mt
                state['v'] = vt
                state['t'] = t
                self.state[p] = state


        return loss
