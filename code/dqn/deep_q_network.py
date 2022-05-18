import os
import numpy as np 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self,lr,n_actions,name,input_dims,chkpt_dir):
        super(DeepQNetwork,self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name)
        
        self.fc1 = nn.Linear(*input_dims,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,n_actions)
        
        self.optimizer =optim.RMSprop(self.parameters(),lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self,state):
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        actions = self.fc3(layer2)
        return actions
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(),self.checkpoint_file)
    
    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))