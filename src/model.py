import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
class Policy(nn.Module):
    def __init__(self,n_regions):
        """
        Initialize the neural network

        Parameters
        ----------
        n_regions : int
            The number of regions in the environment
        
        Returns
        ----------
        None
        """
        super().__init__()
        self.n_regions = n_regions
        self.hidden=nn.Sequential(nn.Linear(3*self.n_regions,512),nn.Tanh(),nn.Linear(512,95))
        self.hidden_x=nn.Sequential(nn.Linear(3,15),nn.Tanh(),nn.Linear(15,5))
        self.out=nn.Sequential(nn.Tanh(),nn.Linear(100,400),nn.Tanh(),nn.Linear(400,2))
    def forward(self, states, goals):
        """
        Compute the forward pass of the neural network

        Parameters
        ----------
        states : torch.tensor
        goals : torch.tensor
        
        Returns
        ----------
        logits : torch.tensor
        """
        x = torch.cat([states, goals.unsqueeze(-1)], dim=-1)
        hid_x=self.hidden_x(x)
        hid=self.hidden(x.reshape(x.shape[0],-1))
        return self.out(torch.cat([hid_x,hid.unsqueeze(1).repeat(1,self.n_regions,1)],dim=-1))
    def act(self,states,goals,action=None):
        """
        Make an action

        Parameters
        ----------
        states : torch.tensor
        goals : torch.tensor
        action : torch.tensor or None

        Returns
        ----------
        action : torch.tensor
        logprobs : torch.tensor
            The log probabilities of actions
        entropy : torch.tensor
            The entropy of the distribution
        """
        logits=self(states,goals)
        probs=Categorical(logits=logits)
        if action is None:
            action=probs.sample()
        logprobs=probs.log_prob(action)
        entropy=probs.entropy()
        return action,logprobs,entropy
    

class Critic(nn.Module):
    def __init__(self,n_regions):
        """
        Initialize the neural network

        Parameters
        ----------
        n_regions : int
            The number of regions in the environment
        
        Returns
        ----------
        None
        """
        super().__init__()
        self.n_regions = n_regions
        self.hidden=nn.Sequential(nn.Linear(3*self.n_regions,512),nn.Tanh(),nn.Linear(512,512),nn.Tanh(),nn.Linear(512,1))
    def forward(self, states, goals):
        """
        Compute the forward pass of the neural network

        Parameters
        ----------
        states : torch.tensor
        goals : torch.tensor
        
        Returns
        ----------
        logits : torch.tensor
        """
        x = torch.cat([states, goals.unsqueeze(-1)], dim=-1)
        
        hid=self.hidden(x.reshape(x.shape[0],-1))
        return hid
        
