import numpy as np
class Env(object):
    def __init__(self,batch_size,complex_sampling=False):
        """
        Initialize the environment.
        Make sure that batch size>=2 for complex sampling to work properly (if enabled) when using reset. 
        The environment is vectorized.

        Parameters
        ----------
        batch size : int
            The size of the batch
        complex sampling : bool
            Whether to use complex sampling or not.
        Returns
        ----------
        None
        """
        self.batch_size=batch_size
        self.complex_sampling=complex_sampling
        self.epsilon=0.005
        self.regions=np.load("./data/regions.npy",allow_pickle=True)
        self.n_regions=len(self.regions)
        self.U=np.load("./data/U.npy")
        self.curr_U=self.U
        self.b=np.load("./data/b.npy")
        self.g=np.load("./data/g.npy")
        self.SR=np.ones((self.n_regions,self.n_regions))-np.eye(self.n_regions)
        self.p=np.load("./data/p.npy")
        self.p/=self.p.sum()
    def reset_curr(self,goal=None):
        """
        Reset the environment to current state in the country given by data

        Parameters
        ----------
        goal : np.ndarray
            The goal to reach

        Returns
        ----------
        state : np.ndarray
        goal : np.ndarray
        """
        self.obs=self.p[None]
        self.curr_U=self.U.copy()[None]
        if goal is None:
            self.goal=np.random.dirichlet(np.ones(self.n_regions))[None]
        else:
            self.goal=goal
        return np.concatenate((self.obs[:,:,None],self.curr_U[:,:,None]),axis=-1),self.goal
    def reset(self):
        """
        Reset the environment to some random state and goal. Is useless at inference

        Returns
        ----------
        state : np.ndarray
        goal : np.ndarray
        """
        self.obs=np.random.dirichlet(np.ones(self.n_regions),size=self.batch_size)
        self.curr_U=self.U+np.random.normal(0,1,(self.batch_size,self.n_regions))
        self.goal=np.zeros((self.batch_size,self.n_regions))
        if self.complex_sampling==True:

            self.goal[:self.batch_size//2]=np.random.dirichlet(np.ones(self.n_regions),size=self.batch_size//2)
            for i in range(self.batch_size//2):
                self.goal[self.batch_size//2+i]=np.random.dirichlet(np.exp(np.random.normal(2,2,self.n_regions)))
        else:
            self.goal[:]=np.random.dirichlet(np.ones(self.n_regions),size=self.batch_size)
        return np.concatenate((self.obs[:,:,None],self.curr_U[:,:,None]),axis=-1),self.goal
    def step(self,action):
        """
        Perform a step in the environment

        Parameters
        ----------
        action : np.ndarray
            The action to perform

        Returns
        ----------
        state : np.ndarray
        reward : np.ndarray
        done : np.ndarray
        info : np.ndarray
            The info is just the error term between the goal and the distribution of labour force among regions
        """
        self.curr_U=self.curr_U+(action*2-1)*0.05
        Umesh0=np.reshape(self.curr_U[:,None,:].repeat(self.n_regions,axis=1),(self.batch_size,-1,))
        Umesh1=np.reshape(self.curr_U[:,:,None].repeat(self.n_regions,axis=2),(self.batch_size,-1,))
        
        U_diff=Umesh0-Umesh1
        U_matrix=U_diff.reshape(self.batch_size,self.n_regions,self.n_regions)
        U_matrix=np.exp(U_matrix+self.b*self.SR)
        U_matrix/=U_matrix.sum(axis=-1,keepdims=True)
        self.obs=np.einsum("br,brn->bn",self.obs,U_matrix)+self.g*self.obs
        self.obs/=self.obs.sum(axis=-1,keepdims=True)
        done=np.array([False]*self.batch_size)
        error=((self.goal-self.obs)**2).sum(axis=-1)**(1/2)
        reward=np.where(error<self.epsilon,0,-1)
        return np.concatenate((self.obs[:,:,None],self.curr_U[:,:,None]),axis=-1),reward,done,error