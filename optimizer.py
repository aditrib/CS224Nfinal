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

                if len(state) == 0:
                    state['t'] = 0
                    state['mt'] =  torch.zeros_like(p.data)
                    state['vt'] =  torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                state['t'] += 1 

                # moving average 1st and 2nd moments
                state['mt'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['vt'].mul_(beta2).add_((1 - beta2) * (grad * grad))

                # bias corrections for 'warmup' effect
                alphat = alpha * math.sqrt(1 - beta2 ** state['t']) / (1 - beta1 ** state['t'])

                # actually take a step for params
                p.data.addcdiv_(state['mt'], 
                                state['vt'].sqrt().add_(eps), 
                                value=-alphat)

                # L2 reg
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-alpha * weight_decay)

        return loss