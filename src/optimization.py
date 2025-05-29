import torch
from torch.optim import Adam
import random
from src import Env
from src import Policy, Critic
import os
import time 
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
class PPO(object):
    def __init__(self,tensorboard=True,pretrain=True,complex_sampling=False,lr=1e-4,batch_size=256,seed=42) -> None:
        """
        Initialize the algorithm

        Parameters
        ----------
        tensorboard : bool
            Turn on logging in tensorboard
        pretrain : bool
            Turn on the pretrain mode of the algorithm
        complex_sampling : bool
            Turn on the complex sampling
        lr : float
            The learning rate
        batch_size : int
            The batch size
        seed : int or None
            The seed
        
        Returns
        ----------
        None
        """
        self.seed_everything(seed)
        self.tensorboard=tensorboard
        self.lr=lr
        self.pretrain=pretrain
        self.i=0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.complex_sampling=complex_sampling
        self.batch_size=batch_size
        self.minibatch=5120
        self.update_epochs=4
        self.obs_dim=2
        self.env=Env(self.batch_size,self.complex_sampling)
        self.steps=400
        self.n_regions=self.env.n_regions
        self.goals=torch.zeros((self.steps,self.batch_size,self.n_regions)).to(self.device)
        self.obs=torch.zeros((self.steps,self.batch_size,self.n_regions,self.obs_dim)).to(self.device)
        self.actions=torch.zeros((self.steps,self.batch_size,self.n_regions)).to(self.device)
        self.logprobs=torch.zeros((self.steps,self.batch_size,self.n_regions)).to(self.device)
        self.rewards =torch.zeros((self.steps,self.batch_size)).to(self.device)
        self.values = torch.zeros((self.steps,self.batch_size)).to(self.device)
        self.advantages = torch.zeros((self.steps,self.batch_size)).to(self.device)
        self.dones = torch.zeros((self.steps,self.batch_size)).to(self.device)
        self.actor=Policy(self.n_regions).to(self.device)
        self.critic=Critic(self.n_regions).to(self.device)
        os.chdir("chkpts")
        valid_files=sorted([file for file in os.listdir() if file.startswith("whole")])
        if valid_files!=[]:
            model=torch.load(valid_files[-1])
            self.actor.load_state_dict(model["actor"])
            self.critic.load_state_dict(model["critic"])
        os.chdir("..")
        self.optimizer_actor= Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic= Adam(self.critic.parameters(), lr=1e-3)
        if self.tensorboard==True:
            self.writer = SummaryWriter("runs/{}".format(time.strftime("%Y.%m.%d.%H.%M.%S")))
    def seed_everything(self,seed):
        """
        Make the experiments reproducible 

        Parameters
        ----------
        seed: int
        
        Returns
        ----------
        None
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
    def simulate(self,obs,goal):
        """
        Use expert to make action

        Parameters
        ----------
        obs : torch.tensor
            The observation
        goal : torch.tensor
            The goal

        Returns
        ----------
        action : torch.tensor
        """
        actions=(torch.where(goal-obs!=0,torch.sign(goal-obs),-1)+1)//2
        return actions
    def rollout(self):
        """
        Make a batched rollout 
        """
        obsie,goal=self.env.reset()
        obs=torch.from_numpy(obsie).to(self.device)
        goals=torch.from_numpy(goal).to(self.device)
        self.goals[:]=goals[None].repeat(self.steps,1,1).to(self.device)
        for step in trange(self.steps):
            self.obs[step]=obs
            with torch.no_grad():
                if self.pretrain==True:
                    actions=self.simulate(self.obs[step,:,:,0],self.goals[step])
                    actions_policy=self.actor.act(self.obs[step],self.goals[step],actions)
                else:
                    actions_policy=self.actor.act(self.obs[step],self.goals[step])
                self.actions[step]=actions_policy[0].squeeze()
                self.values[step]=self.critic(self.obs[step],self.goals[step]).squeeze()
                self.logprobs[step]=actions_policy[1].squeeze()
            obsie,reward,done,info=self.env.step(actions_policy[0].squeeze().cpu().numpy())
            obs=torch.from_numpy(obsie).to(self.device)
            rewards=torch.from_numpy(reward).to(self.device)
            self.rewards[step]=rewards
    def compute_adv_returns(self):
        """
        Compute the advantage and returns for PPO
        """
        with torch.no_grad():
            next_value= self.critic(self.obs[-1],self.goals[-1]).squeeze()
            lastgaelam = 0
            for t in reversed(range(self.steps)):
                if t == self.steps - 1:
                    nextnonterminal = torch.ones((self.batch_size)).to(self.device)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues=self.values[t + 1]
                delta = self.rewards[t] + 0.99 * nextvalues* nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal* lastgaelam
            self.returns = self.advantages + self.values
    def update_ppo(self):
        """
        Perform the learning step
        """
        b_obs = self.obs.reshape((-1,self.n_regions,self.obs_dim))
        b_logprobs = self.logprobs.reshape((-1,self.n_regions))
        b_goals=self.goals.reshape((-1,self.n_regions))
        b_actions= self.actions.reshape((-1,self.n_regions))
        b_advantages = self.advantages.reshape((-1))
        b_returns = self.returns.reshape((-1))
        b_values = self.values.reshape((-1))
        b_inds = np.arange(self.batch_size*self.steps)
        clipfracs=[]
        pg_losses=[]
        e_losses=[]
        v_losses=[]
        if self.pretrain==True:
            ce_losses=[]
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size*self.steps, self.minibatch):
                end = start + self.minibatch
                mb_inds = b_inds[start:end]
                
                action = self.actor.act(b_obs[mb_inds],b_goals[mb_inds],b_actions[mb_inds])
                newvalue=self.critic(b_obs[mb_inds],b_goals[mb_inds])
                logratio = action[1] - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl= (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > 0.2).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                
                mb_advantages = ((mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)).unsqueeze(-1)
                
                # Policy loss
                pg_loss1= mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                pg_loss = (-torch.min(pg_loss1, pg_loss2)).mean()

                
                # Value loss
                newvalue = newvalue.squeeze()
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * v_loss_unclipped.mean()
                v_losses.append(v_loss.item())

                entropy_loss = action[2].mean()
                pg_loss=pg_loss
                e_loss=entropy_loss
                loss = pg_loss -0.01*e_loss
                pg_losses.append(pg_loss.item())
                e_losses.append(e_loss.item())
                
                if self.pretrain==True:
                    self.optimizer_actor.zero_grad()
                    logits=self.actor(b_obs[mb_inds],b_goals[mb_inds]).permute((0,2,1))
                    loss=F.cross_entropy(logits,b_actions[mb_inds].long()).mean()
                    loss.backward()
                    self.optimizer_actor.step()
                    ce_losses.append(loss.item())
                else:

                    self.optimizer_actor.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(),1)
                    self.optimizer_actor.step()

                self.optimizer_actor.zero_grad()
                logits=self.actor(b_obs[mb_inds],b_goals[mb_inds]).permute((0,2,1))
                loss=F.cross_entropy(logits,b_actions[mb_inds].long())
                loss.mean().backward()
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                v_loss.backward()
                self.optimizer_critic.step()
                
        if self.tensorboard==True:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true  - y_pred) / var_y
            self.writer.add_scalar("losses_critic/value_loss", np.mean(v_losses), self.i)
            
            if self.pretrain==True:
                self.writer.add_scalar("losses/ce_loss", np.mean(ce_losses), self.i)
            
            self.writer.add_scalar("losses/policy_loss", np.mean(pg_losses), self.i)
            
            self.writer.add_scalar("entropy/entropy", np.mean(e_losses), self.i)

            self.writer.add_scalar("approx_kl/old_approx_kl", old_approx_kl.item(), self.i)
            self.writer.add_scalar("approx_kl/approx_kl", approx_kl.item(), self.i)
            self.writer.add_scalar("approx_kl/clipfrac", np.mean(clipfracs), self.i)


            self.writer.add_scalar("explained_variance/explained_variance", explained_var, self.i)
            
            self.writer.add_scalar("rewards/rewards", self.rewards.sum(dim=0).mean(), self.i)
            
    def cycle(self,epochs):
        """
        Perform the learning cycle for a given number of epochs

        Parameters
        ----------
        epochs : int
        
        Returns
        ----------
        None
        """
        for i in range(epochs):
            print(i)
            self.rollout()
            self.compute_adv_returns()
            self.update_ppo()
            self.i+=1
            if i%10==0:
                savedict={"actor":self.actor.state_dict(),
                          "critic":self.critic.state_dict()}
                torch.save(savedict,"./chkpts/whole_{}.pt".format(time.strftime("%Y.%m.%d.%H.%M.%S")))
