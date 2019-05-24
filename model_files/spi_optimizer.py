import torch
from torch.optim.optimizer import Optimizer, required
import numpy
#from .optimizer import Optimizer, required


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,KD=0.,KDD=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,KD=KD,KDD=KDD)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            KD = group['KD'] 
            KDD =group['KDD']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    flag=0
                    if 'g_buffer' not in param_state:
                        g_buf = param_state['g_buffer'] = torch.zeros_like(p.data)
                        gg_buf = torch.zeros_like(p.data)
                    else:
                        g_buf = param_state['g_buffer']
                        if'pre_grad_buffer' not in param_state:
                            gg_buf = param_state['pre_grad_buffer'] = torch.zeros_like(p.data)
                            gg_buf.add_(g_buf)#d_p

                        else:
                            gg_buf = param_state['pre_grad_buffer']#d_p
                            # KDD_buf=d_p-2*g_buf+gg_buf
                            flag=1
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)#1:0,0
                        if flag<1:
                            KDD=0
                        else:
                            KDD =group['KDD']
                    else:
                        buf = param_state['momentum_buffer']
                        sign=torch.sign(d_p) * torch.sign(buf)
                        # sign=sign.cpu().numpy()
                        # print(sign)
                        # sign_array=sign.flatten()
                        # sign_array.nonzero
                        # for index, value in enumerate(sign_array):
                        #     if value == 0:
                        #         print(index,',',end='')
                        # # index=sign_array.index(0)
                        # index = numpy.argwhere(sign_array == 0)
                        # if index.any():
                        #     print(index, ',', end='')
                        # sign=numpy.clip(sign,-1,0)
                        # sign=torch.from_numpy(sign)
                        # sign=sign.cuda()
                        # T=
                        buf_value=torch.zeros_like(p.data)#gai
                        buf_value.add_(buf)#gai
                        buf.mul_(0.5*sign).add_(0.5*buf_value)#gai
                        # buf.mul_(0.5)

                        if flag<1:
                            KDD=0
                        else:
                            KDD =group['KDD']

                        buf.mul_(momentum).add_(1 - dampening, d_p)

                    if nesterov:
                        d_p2 = d_p.add(momentum, buf)
                    else:
                        d_p2 = buf

                p.data.add_(-group['lr'], d_p2).add_(-group['lr']*KD,(d_p-g_buf)).add_(-group['lr']*KDD,d_p-2*g_buf+gg_buf)#d_p,d_p
                gg_buf.add_(-gg_buf).add_(g_buf.clone())  # 0,d_p
                g_buf.add_(-g_buf).add_(d_p.clone())  # d_p,d_p2

        return loss
