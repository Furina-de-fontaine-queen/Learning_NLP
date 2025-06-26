import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset
from tqdm import tqdm
import os 
import sys
from torch.utils.tensorboard import SummaryWriter
class Mytrainer: 
    def __init__(self,
                 train_dataset:Dataset,
                 val_dataset:Dataset,
                 model:nn.Module,
                 num_classes:int=5,
                 epochs:int=10,
                 batch_size:int=8,
                 lr:float=0.001,
                 optimizer = optim.Adam,
                 weights:str='',
                 device = 'cuda:0'):
        self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
        self.optimizer = optimizer(model.parameters(),lr=lr)
        self.model = model 
        self.num_classes = num_classes
        self.epochs = epochs 
        self.device = device 

    def train(self):

        if os.path.exists('./weights') is False:
            os.makedirs('./weights')
        tb_writer = SummaryWriter()
        for epo in range(self.epochs): 
            train_loss, train_acc = train_one_epoch(model=self.model,
                                                    optimizer=self.optimizer,
                                                    data_loader=self.train_dataloader,
                                                    device=self.device,
                                                    epoch=epo)
            
            val_loss, val_acc = evaluate(model=self.model,
                                         data_loader=self.val_dataloader,
                                         device=self.device,
                                         epoch=epo)
            
            tags = ['train_loss','train_acc','val_loss','val_acc']

            tb_writer.add_scalar(tags[0],train_loss,epo)
            tb_writer.add_scalar(tags[1],train_acc,epo)
            tb_writer.add_scalar(tags[2],val_loss,epo)
            tb_writer.add_scalar(tags[3],val_acc,epo)

            torch.save(self.model.state_dict,'./weights/model-{}.pth'.format(epo))


def train_one_epoch(model,optimizer,data_loader,device,epoch):
    model.train() 
    loss_func = nn.CrossEntropyLoss() 
    acc_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader,file=sys.stdout)

    for step, data in enumerate(data_loader): 
        image, label = data 
        image = image.reshape(-1,28 * 28)
        sample_num += image.shape[0]

        pred = model(image.to(device))
        pred_cls = torch.max(pred,dim=1)[1]
        acc_num += torch.eq(pred_cls,label.to(device)).sum()

        loss = loss_func(pred,label.to(device))
        loss.backward() 
        acc_loss += loss.detach()

        data_loader.desc = f'[train epoch {epoch}], loss: {acc_loss.item() / (step + 1):.3f}, acc_num: {acc_num.item() / sample_num:.3f}'

        if not torch.isfinite(loss):
            print('WARNING:non infinite loss,ending training',loss)
            sys.exit(1)

        optimizer.step() 
        optimizer.zero_grad()

    return acc_loss.item() / (step + 1), acc_num.item() / sample_num

@torch.no_grad() 
def evaluate(model,data_loader,device,epoch):
    loss_func = nn.CrossEntropyLoss() 

    model.eval() 
    acc_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)
    
    sample_num = 0 
    data_loader = tqdm(data_loader,file=sys.stdout)

    for step, data in enumerate(data_loader):
        image, label = data 
        image = image.reshape(-1,28*28)
        sample_num += image.shape[0]

        pred = model(image.to(device))
        pred_cls = torch.max(pred,dim=1)[1]
        acc_num += torch.eq(pred_cls,label.to(device)).sum()

        loss = loss_func(pred,label.to(device))
        acc_loss += loss.detach()

        data_loader.desc = f'[train epoch {epoch}], loss: {acc_loss.item() / (step + 1):.3f}, acc_num: {acc_num.item() / sample_num: .3f}'

    return acc_loss.item() / (step + 1) , acc_num.item() / sample_num




