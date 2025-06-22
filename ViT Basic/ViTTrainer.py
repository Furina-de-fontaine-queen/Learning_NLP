from PIL import Image 
import torch 
from torch.utils.data import Dataset
import os 
import sys 
import json 
import pickle 
import random
import math
from MyViT import ViT
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
class MyDataSet(Dataset):
    r'''custom dataset
    '''

    def __init__(self,images_path:list,image_class:list,transform=None):
        self.images_path = images_path
        self.image_class = image_class 
        self.transform = transform 

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        image =  Image.open(self.images_path[index])

        if image.mode != 'RGB':
            raise ValueError(f'image:{self.images_path[index]} is not RGB mode.')
        label = self.image_class[index]

        if self.transform is not None:
            image = self.transform(image)

        return image,label 
    

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images,dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    
class MyTrainer:
    def __init__(self,num_classes:int=5,
                 epochs:int=10,
                 batch_size:int=8,
                 lr:float=0.001,
                 lrf:float=0.01,
                 data_path:str='/data/flower_photos',
                 model_name:str='',
                 weights:str='./vit_base_patch16_224_in21k.pth',
                 freeze_layers:bool=True,
                 device = 'cuda:0'):
        self.num_classes = num_classes 
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.lr = lr 
        self.lrf = lrf 
        self.data_path = data_path 
        self.model_name = model_name 
        self.weights = weights 
        self.freeze_layers = freeze_layers 
        self.device = device 

    def train(self):
        device = torch.device(self.device)

        if os.path.exists('./weights') is False:
            os.makedirs('./weights')

        tb_writer = SummaryWriter()

        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(self.data_path)
        # transform provides dedicated module for image preprocessing ,which provides:
        # Various image transformations
        # Data Augumentation methods 
        # Tensor comvertion and normalization tools

        # Compose can link all transformations to a pipeline
        # RandomResizedCrop(m): randomly cropping + resizing
        # RandomHorizontalFlip(): random flip the image horizonly
        # Normalize(): norm_img = (img-mean) / std
        data_transform = {
            'train':transforms.transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]),
            'val': transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                       transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
        }

        train_dataset = MyDataSet(images_path=train_images_path,
                                  image_class=train_images_label,
                                  transform=data_transform['train'])
        val_dataset = MyDataSet(images_path=val_images_path,
                                image_class=val_images_label,
                                transform=data_transform['val'])
        
        batch_size = self.batch_size 
        num_work = min([os.cpu_count(), batch_size if batch_size > 1 else 0,8])
        print(f'Using {num_work} dataloader workers every process.')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=num_work,
                                                   collate_fn=train_dataset.collate_fn)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=num_work,
                                                 collate_fn=val_dataset.collate_fn)
        
        model = ViT(img_size=224,patchsize=16,embed_dim=768,depth=12,num_heads=12,representation_size=None,num_classes=self.num_classes).to(device)

        if self.weights != '':
            assert os.path.exists(self.weights), f'weight file: {self.weights} not exist.'
            weights_dict = torch.load(self.weights,map_location=device)

            del_keys = ['head.weight','head.bias'] if model.has_logits else ['pre_logits.fc.weight','pre_logits.fc.bias','head.weight','head.bias']
            for k in del_keys:
                del weights_dict[k] 
            print(model.load_state_dict(weights_dict,strict=False))

        # only training head's parameters
        if self.freeze_layers:
            for name , para in model.named_parameters():
                if 'head' not in name and 'pre_logits' not in name: 
                    para.requires_grad_(False)
                else:
                    print(f'training {name}')

        pg = [p for p in model.parameters() if p.requires_grad] # gather up the trainable params list
        optimizer = optim.SGD(pg,lr=self.lr,momentum=0.9,weight_decay=5e-5)  

        lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - self.lrf) + self.lrf 
        scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda=lf)  # Custom lr adjustment

        for epoch in range(self.epochs):
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)
            scheduler.step()

            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)
            
            tags = ['train_loss','train_acc','val_loss','val_acc','learning_rate']
            tb_writer.add_scalar(tags[0], train_loss,epoch)
            tb_writer.add_scalar(tags[1], train_acc,epoch)
            tb_writer.add_scalar(tags[2], val_loss,epoch)
            tb_writer.add_scalar(tags[3], val_acc,epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'],epoch)
            
            torch.save(model.state_dict(), './weights/model-{}.pth'.format(epoch))
        
def read_split_data(root:str,val_rate:float = 0.2):
    r'''
    Read the paths of of each class from folders in root class and visulize the distribution of each class
    '''
    random.seed(0)
    assert os.path.exists(root), f'dataset root: {root} does not exist.'

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root,cla))]
    flower_class.sort() 

    # give each class a digit index
    cls_indices = dict((k,v) for v,k in enumerate(flower_class))
    json_str = json.dumps(dict((val,key) for key, val in cls_indices.items()),indent=4)

    with open('cls_indices.json','w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_cls_num = [] 
    supported = ['.jpg','.JPG','.png','.PNG']


    # Load and split thr dataset for each path
    for cla in flower_class:
        cla_path = os.path.join(root,cla)
        images = [os.path.join(root,cla,i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_cls = cls_indices[cla]
        every_cls_num.append(len(images))
        val_path = random.sample(images,k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_cls)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_cls)
    assert (len(train_images_path) > 0) and (len(val_images_path) > 0) 
    print(f'{sum(every_cls_num)} were found in the dataset.')
    print(f'{len(train_images_path)} images for training.')
    print(f'{len(val_images_path)} images for valiadation')

    plot_image = False 
    if plot_image:
        plt.bar(range(len(flower_class)), every_cls_num, align='center')
        plt.xticks(range(len(flower_class)),flower_class)

        for i,v in enumerate(every_cls_num):
            plt.text(x=i,y=v + 5,s = str(v),ha='center')

        plt.xlabel('image class')
        plt.ylabel('number of images')

        plt.title('flower class distribution')
        plt.show() 

    return train_images_path, train_images_label, val_images_path, val_images_label


def train_one_epoch(model,optimizer,data_loader,device,epoch):
    model.train()
    loss_func = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad() 

    sample_num = 0
    data_loader = tqdm(data_loader,file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data 
        sample_num += images.shape[0]

        pred = model(images.to(device))  # pred: [batch_size, num_classes]
        pred_classes = torch.max(pred,dim=1)[1] # torch.max() will return (max_val,argmax_val)
        accu_num += torch.eq(pred_classes,labels.to(device)).sum() 

        # backward to update params
        loss = loss_func(pred,labels.to(device))
        loss.backward()
        accu_loss += loss.detach() # convert torch.type to value 

        data_loader.desc = f'[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num}'

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

        optimizer.step() 
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model,data_loader,device,epoch):
    loss_func = torch.nn.CrossEntropyLoss() 

    model.eval() 

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader,file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data 
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred,dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum() 

        loss = loss_func(pred, labels.to(device))
        accu_loss += loss 

        data_loader.desc = f'[valid epoch {epoch} loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num}]'

    return accu_loss.item() / (step + 1) , accu_num.item() / sample_num