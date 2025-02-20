import torch
import torch.nn as nn
from utils import uniform
from deq_lib.solvers import broyden
from activations import Tanh


class RENLayer(nn.Module):
    """
    Recurrent Equilibrium Network (REN) layer with implicit function to recover w(k).
    
    x(k+1) = A*x(k) + B1*w(k) + B2*u(k)
    y(k) = C2*x(k) + D21*w(k) + D22*u(k)
    w(k) = tanh(C1*x(k) + D11*w(k) + D12*u(k))
    """
    def __init__(self,future_steps, x_size, y_size, u_size, w_size, solver=broyden, f_thresh=30, b_thresh=30):
        super(RENLayer, self).__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.u_size = u_size
        self.w_size = w_size
        self.hook = None
        self.future_steps = future_steps
        # Learnable weights
        self.A = nn.Parameter(uniform(x_size, x_size))
        self.B1 = nn.Parameter(uniform(w_size, x_size))
        self.B2 = nn.Parameter(uniform(u_size, x_size,))
        self.C1 = nn.Parameter(uniform(x_size,w_size))
        self.D11 = nn.Parameter(uniform(w_size, w_size))
        
        self.D12 = nn.Parameter(uniform(u_size,w_size))
        self.C2 = nn.Parameter(uniform(x_size,y_size))
        self.D21 = nn.Parameter(uniform(w_size,y_size))
        self.D22 = nn.Parameter(uniform(u_size,y_size))

        self.solver = solver
        self.f_thresh = f_thresh
        self.b_thresh = b_thresh
        self.hook = None
        self.delta = Tanh()

    def forward(self, x, u):
        
        device = u.device

        batch_size = u.shape[0] 
        #u = torch.reshape(u, (batch_size,1,-1))
        
        w0 = torch.zeros(batch_size, 1, self.w_size).to(device)

        # Solve for w(k) implicitly
        with torch.no_grad():
            w_star = self.solver(
                lambda w: self.delta(x @ self.C1 + w @ self.D11 + u @ self.D12 ),
                w0,
                threshold=self.f_thresh
            )['result']
            new_w_star = w_star
        
        if self.training:
            w_star.requires_grad_()
            new_w_star = self.delta(x @ self.C1 + w_star @ self.D11 + u @ self.D12)

            def backward_hook(grad):
                
                if self.hook is not None:
                    self.hook.remove()
                    
                new_grad = self.solver(
                    lambda g: torch.autograd.grad(new_w_star, w_star, g, retain_graph=True)[0] + grad,
                    torch.zeros_like(grad).to(device),
                    threshold=self.b_thresh
                )['result']
                return new_grad

            self.hook = new_w_star.register_hook(backward_hook)

        #new_w_star = (x @ self.C1 + u @ self.D12)
        # Compute next state x(k+1)
        x_next = (x @ self.A + new_w_star @ self.B1 + u @ self.B2)
        
        # Compute output y(k)
        y = (x @ self.C2 + new_w_star @ self.D21 + u @ self.D22)#.view(batch_size, self.y_size)
        
        return x_next, y

# Example usage
# ren_layer = RENLayer(x_size=10, y_size=5, u_size=3, w_size=7)
# x = torch.randn(4, 10)
# u = torch.randn(4, 3)
# x_next, y = ren_layer(x, u)
