from typing import List
import torch
from torch.autograd import grad
import torch.nn as nn
from torch.nn import functional as F
import numpy as np 
from scipy import ndimage
from torch.nn.init import xavier_normal_
import time, os
import scipy
from utils import solve_lap
from torch.autograd import Variable
from  torch.optim.lr_scheduler import ExponentialLR

EPS=1e-6


class CycleLoss(nn.Module):

    def __init__(self, loss_type = "L1"):
        super(CycleLoss, self).__init__()
        if loss_type == "L1":
            loss = nn.L1Loss()
            self.loss = loss
        elif loss_type == "L2":
            loss = nn.L2Loss()
            self.loss = loss

    def forward(self, x12: torch.Tensor, x23: torch.Tensor, x31: torch.Tensor):
        assert(x12.shape==x23.shape)
        assert(x23.shape==x31.shape)
        cycle = torch.matmul(torch.matmul(x12,x23),x31)
        eye = torch.eye(x12.shape[0]).to(cycle.device)
        return self.loss(cycle, eye)



class LAPSolver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cost: torch.Tensor, params:dict):
        device = cost.device
        labelling = torch.zeros_like(cost)
        cost_np = cost.cpu().detach().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_np)
        labelling[row_ind, col_ind] = 1.
        ctx.labels = labelling
        ctx.label_inds = col_ind
        ctx.params = params
        ctx.cost = cost
        ctx.device = device
        return labelling.to(device)

    @staticmethod
    def backward(ctx, unary_gradients: torch.Tensor):
        """
        :param ctx: context for backpropagation. Gradient is stored here.
        :param unaries: List
        :param pwCost: dictionary
        """
        assert(ctx.cost.shape==unary_gradients.shape)

        device=unary_gradients.device
        lambda_val = ctx.params["lambda"]
        fwd_labels = ctx.labels
        unaries = ctx.cost

        unaries_prime=unaries + lambda_val*unary_gradients 
        bwd_labels, label_inds = solve_lap(unaries_prime)
        unary_grad_bwd = (bwd_labels-fwd_labels) / (lambda_val + EPS)


        return unary_grad_bwd.to(device), None

class LAPCycleSolverModule(torch.nn.Module):

    def __init__(self,params_dict):
        super(LAPCycleSolverModule, self).__init__()
        self.params_dict=params_dict
        self.params_dict["const"]=100   #   constant to subtract from the qap solver 
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=EPS)
        self.lapSolver = LAPSolver()

    def forward(self,un1: torch.Tensor, un2:torch.Tensor, un3: torch.Tensor):

        n_un1=nn.functional.normalize(un1, p=2, dim=1)
        n_un2=nn.functional.normalize(un2, p=2, dim=1)
        n_un3=nn.functional.normalize(un3, p=2, dim=1)

        c12 = 1 - torch.mm(n_un1, n_un2.t())
        c23 = 1 - torch.mm(n_un2, n_un3.t())
        c31 = 1 - torch.mm(n_un3, n_un1.t())


        x12=self.lapSolver.apply(c12,self.params_dict)
        x23=self.lapSolver.apply(c23,self.params_dict)
        x31=self.lapSolver.apply(c31,self.params_dict)
        return x12, x23, x31


if __name__ == "__main__":

    torch.manual_seed(66)
    #   Let U is the set of unaries. H is the set of hypotheses.
    num_unaries, num_hypotheses = 4, 4  #15, 15         #   Set the num_unaries |U| and num_hypotheses |H|

    un12 = torch.randint(0,10,(num_unaries,num_hypotheses)).to(float).cuda()
    un23 = torch.randint(0,10,(num_unaries,num_hypotheses)).to(float).cuda()
    un31 = torch.randint(0,10,(num_unaries,num_hypotheses)).to(float).cuda()

    un12 = Variable(un12, requires_grad=True)
    un23 = Variable(un23, requires_grad=True)
    un31 = Variable(un31, requires_grad=True)


    params={'lambda':70,'costMargin':0.4}
    lap_solver = LAPCycleSolverModule(params)
    
    # optimizer=torch.optim.SGD([unary_costs, pw_costs],lr=1e-3,momentum=0.9)
    optimizer = torch.optim.Adam([un12, un23, un31],lr=1e-1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    min_loss=100.
    epochs = 10000
    cost_margin = 0.5
    loss_fn=CycleLoss()
    x12, x23, x31 = lap_solver(un12,un23,un31)
    loss = loss_fn(x12,x23,x31)
    print(f"loss.item(): {loss.item()}")
    prev_unaries, prev_pairwise = 0., 0.

    

    for ep in range(epochs):
        optimizer.zero_grad()
        x12, x23, x31 = lap_solver(un12,un23,un31)

        loss = loss_fn(x12, x23, x31)

        print(f"loss.item(): {loss.item()}")

        loss.backward()
        optimizer.step()
        scheduler.step()

        input()