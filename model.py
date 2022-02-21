import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size).cpu()
        self.linear2 = nn.Linear(hidden_size,output_size).cpu()
        
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    def save(self, file_name='model.pth'):
        # model_folder_path = 'D:\SnakeAI'
        model_folder_path = '../'
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimer = optim.Adam(model.parameters(),lr = self.lr)    
        self.criterion = nn.MSELoss()
        # for i in self.model.parameters():
        #     print(i.is_cpu)

    
    # TODO: Explain how this method works.
    """
    We start off by getting the game variables and creating tensors of the
    given data so that we can then use as inputs for our deep Q network model.
    We then check if the state is 1 and if so, we use torch.unsqueeze to convert
    the tensor to a size of one.  After this, we get the predicted q value, and 
    clone it to create the target that we can later use to calculate the loss.
    We then iterate through the done states and update target with the rewards
    from each state, or if the game is ongoing, we use our gamma rate and the
    new state parameters to obtain a new Q value that we store in target using
    the current index.  We then zero out the gradients of the optimized tensors
    and perform a single step in the optimization function.
    """
    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float).cpu()
        next_state = torch.tensor(next_state,dtype=torch.float).cpu()
        action = torch.tensor(action,dtype=torch.long).cpu()
        reward = torch.tensor(reward,dtype=torch.float).cpu()


        if(len(state.shape) == 1): # only one parameter to train , Hence convert to tuple of shape (1, x)
            #(1 , x)
            state = torch.unsqueeze(state,0).cpu()
            next_state = torch.unsqueeze(next_state,0).cpu()
            action = torch.unsqueeze(action,0).cpu()
            reward = torch.unsqueeze(reward,0).cpu()
            done = (done, )

           

        # 1. Predicted Q value with current state
        pred = self.model(state).cpu()
        target = pred.clone().cpu()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).cpu()
            target[idx][torch.argmax(action).item()] = Q_new 
        # 2. Q_new = reward + gamma * max(next_predicted Qvalue) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimer.step()
         

