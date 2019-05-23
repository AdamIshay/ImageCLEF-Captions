
import argparse
import time
import torch
import pickle
import numpy as np

import os


import itertools
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from PIL import Image
import sys


from preprocessing_images import Data,preprocessing
from preprocessing_labels import training_data,validation_data,idx_to_name,idx_to_name_v,idx_to_umls,umls_to_idx


from torch.distributions.multivariate_normal import MultivariateNormal

# Training settings

cuda = torch.cuda.is_available()

seed = 10


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 1000 # y dimensions
z_dim = 200 # vector size for the z vector
train_batch_size = 256
valid_batch_size = 256
N = 1000 #size of hidden layers in discriminator network
epochs = 1 #number of epochs for training

test_F1=False # set to True if you want to test for the F1 score each epoch, this makes training time much longer


params = {'n_classes': n_classes, 'z_dim': z_dim,
          'train_batch_size': train_batch_size,
          'valid_batch_size': valid_batch_size, 'N': N, 'epochs': epochs,
          'cuda': cuda}


use_gpu = True
training_path='training_path'
validation_path='validation_path'



#normalization
mx=0.5;my=0.5;mz=0.5; #mean
sx=.15;sy=.15;sz=.15; #standard deviation


normalize = transforms.Normalize(mean=[mx,my,mz],
                                 std=[sx,sy,sz])

#inv_normalize = transforms.Normalize(mean=[-mx/sx,-my/sy,-mz/sz],std=[1/sx,1/sy,1/sz]) #reverses the normalization

preprocessing = transforms.Compose([
    transforms.Resize((64,64)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

##################################
# Load data and create Data loaders
##################################

def get_loaders(train_labeled_data,train_unlabeled_data,valid_data, labeled_path,unlabeled_path,valid_path):
    ''' returns dataloaders for labeled,unlabeled, and validation sets. The 
    validation and unlabeled are the same here'''
      
        
    training_ds=Data(training_data,directory=training_path,transform=preprocessing)
    
    train_labeled_loader=DataLoader(training_ds,batch_size=train_batch_size,shuffle=True, num_workers=0)
    
    
    unlabeled_ds=Data(validation_data, directory=validation_path,labeled=False,transform=preprocessing)
    
    train_unlabeled_loader=DataLoader(unlabeled_ds,batch_size=train_batch_size,shuffle=True, num_workers=0)
    
    
    validation_ds=Data(validation_data, directory=validation_path,transform=preprocessing)
    
    valid_loader=DataLoader(validation_ds,batch_size=train_batch_size,shuffle=True, num_workers=0)    
    

    return train_labeled_loader, train_unlabeled_loader, valid_loader


##################################
# Define Networks
##################################


# Encoder


class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        
        self.bn=nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(3, 64,kernel_size=3,stride=2,padding=1)
        self.bn1=self.bn(64)
        self.conv2 = nn.Conv2d(64, 128,kernel_size=3,stride=1,padding=1)
        self.bn2=self.bn(128)
        self.conv3 = nn.Conv2d(128, 128,kernel_size=3,stride=2,padding=1)
        self.bn3=self.bn(128)
        self.conv4 = nn.Conv2d(128, 256,kernel_size=3,stride=1,padding=1)
        self.bn4=self.bn(256)
        self.conv5 = nn.Conv2d(256, 256,kernel_size=3,stride=2,padding=1)
        self.bn5=self.bn(256)
        self.conv6 = nn.Conv2d(256, 512,kernel_size=3,stride=1,padding=1)
        self.bn6=self.bn(512)
        self.conv7 = nn.Conv2d(512, 512,kernel_size=3,stride=2,padding=1)
        self.bn7=self.bn(512)
        self.maxpool=nn.MaxPool2d(2)
        # Gaussian code (z)
        self.lin3gauss = nn.Linear(8192, z_dim)
        # Categorical code (y)
        self.lin3cat = nn.Linear(8192, n_classes)

    def forward(self, x):

        x = F.dropout(self.conv1(x), p=0.25, training=self.training)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=.2)
        x = F.dropout(self.conv2(x), p=0.25, training=self.training)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=.2)
        x = F.dropout(self.conv3(x), p=0.25, training=self.training)
        x = self.bn3(x)
        x = F.leaky_relu(x,negative_slope=.2)
        x = F.dropout(self.conv4(x), p=0.25, training=self.training)
        x = self.bn4(x)
        x = F.leaky_relu(x,negative_slope=.2)
        x = F.dropout(self.conv5(x), p=0.25, training=self.training)
        x = self.bn5(x)
        x = F.leaky_relu(x,negative_slope=.2)
        x = F.dropout(self.conv6(x), p=0.25, training=self.training)
        x = self.bn6(x)
        x = F.leaky_relu(x,negative_slope=.2)
        x = F.dropout(self.conv7(x), p=0.25, training=self.training)
        x = self.bn7(x)
        
        xgauss = self.lin3gauss(x.flatten(1))
        xcat = F.sigmoid(self.lin3cat(x.flatten(1))) #should be sigmoid for multi-label

        return xcat, xgauss


# Decoder

class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.bn=nn.BatchNorm2d
        
        self.fc=nn.Linear(z_dim+n_classes,8192)
        self.conv1 = nn.Conv2d(512, 512,kernel_size=3,stride=1,padding=0)
        self.bnc1=self.bn(512)
        self.deconv1 = nn.ConvTranspose2d(512, 512,kernel_size=3,stride=2)
        self.bn1=self.bn(512)
        self.conv2 = nn.Conv2d(512, 256,kernel_size=3,stride=1,padding=0)
        self.bnc2=self.bn(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256,kernel_size=3,stride=2)
        self.bn2=self.bn(256)
        
        
        self.conv3 = nn.Conv2d(256, 128,kernel_size=3,stride=1,padding=1)
        self.bnc3=self.bn(128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 128,kernel_size=3,stride=2)
        self.bn3=self.bn(128)
        
        self.conv4 = nn.Conv2d(128, 64,kernel_size=3,stride=1,padding=2)
        self.bnc4=self.bn(64)
        
        self.deconv4 = nn.ConvTranspose2d(64, 64,kernel_size=3,stride=2,output_padding=1)
        self.bn4=self.bn(64)
        
        self.conv5 = nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=0)
        self.bnc5=self.bn(64)
        
        self.conv6 = nn.Conv2d(64, 3,kernel_size=3,stride=1,padding=0)
        self.bnc6=self.bn(3)
        


    def forward(self, x):
        
        x = self.fc(x)
        
        x=F.dropout(self.deconv1(x.view(x.shape[0],512,4,4)),p=0.25,training=self.training)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=.2)
        x=F.dropout(self.conv2(x),p=0.25,training=self.training)
        x = self.bnc2(x)
        x = F.leaky_relu(x,negative_slope=.2)
        
        x=F.dropout(self.deconv2(x),p=0.25,training=self.training)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=.2)
        
        x=F.dropout(self.conv3(x),p=0.25,training=self.training)
        x = self.bnc3(x)
        x = F.leaky_relu(x,negative_slope=.2)
        
        x=F.dropout(self.deconv3(x),p=0.25,training=self.training)
        x = self.bn3(x)
        x = F.leaky_relu(x,negative_slope=.2)
        
        x=F.dropout(self.conv4(x),p=0.25,training=self.training)
        x = self.bnc4(x)
        x = F.leaky_relu(x,negative_slope=.2)
        
        x=F.dropout(self.deconv4(x),p=0.25,training=self.training)
        x = self.bn4(x)
        x = F.leaky_relu(x,negative_slope=.2)
        x=F.dropout(self.conv5(x),p=0.25,training=self.training)
        x = self.bnc5(x)
        x = F.leaky_relu(x,negative_slope=.2)
        
        x=F.dropout(self.conv6(x),p=0.25,training=self.training)
        x = self.bnc6(x)
        x = F.leaky_relu(x,negative_slope=.2) 
        
        return F.sigmoid(x)


# Discriminator networks
class D_net_cat(nn.Module):
    def __init__(self):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(n_classes, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return F.sigmoid(x)


class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)

        return F.sigmoid(self.lin3(x))


####################
# Utility functions
####################
def sample_categorical(batch_size, n_classes=10):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.autograd.Variable with the sample
    '''
    cat = np.random.randint(0, 10, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)


####################
# Train procedure
####################
def train(P, Q, D_cat, D_gauss, P_decoder, Q_encoder, Q_semi_supervised, Q_generator, D_cat_solver, D_gauss_solver, train_labeled_loader, train_unlabeled_loader):
    '''
    Train procedure for one epoch.
    '''
    TINY = 1e-15

    Q.train()
    P.train()
    D_cat.train()
    D_gauss.train()

    if train_unlabeled_loader is None:
        train_unlabeled_loader = train_labeled_loader

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each

    for (X_l, target_l), (X_u, target_u) in zip(train_labeled_loader, train_unlabeled_loader): #enumerate

        for X, target in [(X_u, target_u), (X_l, target_l)]:

            if target[0].min() == -1:
                labeled = False
            else:
                labeled = True

            # Load batch

            X, target = torch.tensor(X,dtype=torch.float32), torch.tensor(target,dtype=torch.float32)
            if cuda:
                X, target = X.cuda(), target.cuda()

            # Init gradients
            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            #######################
            # Reconstruction phase
            #######################
            if not labeled:
                z_sample = torch.cat(Q(X), 1)# concatenated y and gaussian parameters
                if len(z_sample)!=256:
                    pass;
                X_sample = P(z_sample)
                
                recon_loss = F.binary_cross_entropy(X_sample + TINY, X + TINY)
                recon_loss = recon_loss
                recon_loss.backward()
                P_decoder.step()
                Q_encoder.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()
                recon_loss = recon_loss
                print('recon_loss = '+str(recon_loss.item()))
                #######################
                # Regularization phase
                #######################
                # Discriminator
                Q.eval()
                z_real_cat = sample_categorical(X.shape[0], n_classes=n_classes)
                z_real_gauss = torch.randn(X.shape[0], z_dim) #z_dim will be the dimensional size of gaussian (eg. 500)
                
                if cuda:
                    z_real_cat = z_real_cat.cuda()
                    z_real_gauss = z_real_gauss.cuda()

                z_fake_cat, z_fake_gauss = Q(X)

                D_real_cat = D_cat(z_real_cat)
                D_real_gauss = D_gauss(z_real_gauss)
                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                D_loss_cat = -torch.mean(torch.log(D_real_cat + TINY) + torch.log(1 - D_fake_cat + TINY))
                D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

                D_loss = D_loss_cat + D_loss_gauss
                D_loss = D_loss

                D_loss.backward()
                D_cat_solver.step()
                D_gauss_solver.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()

                # Generator
                Q.train()
                z_fake_cat, z_fake_gauss = Q(X)

                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                G_loss = - torch.mean(torch.log(D_fake_cat + TINY)) - torch.mean(torch.log(D_fake_gauss + TINY))
                G_loss = G_loss
                G_loss.backward()
                Q_generator.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()
                print('recon_loss = '+str(recon_loss.item())+'\t D_loss = '+str(D_loss.item())+'\t G_loss = '+str(G_loss.item()))
            #######################
            # Semi-supervised phase
            #######################
            if labeled:
                pred, _ = Q(X)
                #class_loss = F.cross_entropy(pred, target)
                class_loss = F.binary_cross_entropy(pred, target)
                class_loss.backward()
                Q_semi_supervised.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()
                print('\t\t\t\t\t\t\t\t class_loss = '+str(class_loss.item()))
    return recon_loss,class_loss, D_loss,D_loss_cat, D_loss_gauss, G_loss




def generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader):
    torch.manual_seed(10)
    
    class_loss_hist=[]
    recon_loss_hist=[]
    D_loss_hist=[]
    D_loss_cat_hist=[]
    D_loss_gauss_hist=[]
    G_loss_hist=[]
    
    F1_train_hist=[]
    F1_val_hist=[]
    
    if cuda:
        Q = Q_net().cuda()
        P = P_net().cuda()
        D_cat = D_net_cat().cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net()
        P = P_net()
        D_gauss = D_net_gauss()
        D_cat = D_net_cat()

    # Set learning rates
    gen_lr = 0.0006
    semi_lr = 0.001
    reg_lr = 0.0008

    # Set optimizators
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

    Q_semi_supervised = optim.Adam(Q.parameters(), lr=semi_lr)

    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)
    D_cat_solver = optim.Adam(D_cat.parameters(), lr=reg_lr)

    start = time.time()
    for epoch in range(epochs):

        
        recon_loss,class_loss, D_loss,D_loss_cat, D_loss_gauss, G_loss = train(P, Q, D_cat,
                                                                         D_gauss, P_decoder,
                                                                         Q_encoder, Q_semi_supervised,
                                                                         Q_generator,
                                                                         D_cat_solver, D_gauss_solver,
                                                                         train_labeled_loader,
                                                                         train_unlabeled_loader)

        if epoch % 1 == 0:

            print(epoch)
            print('Train Classification Loss: {:.3}'.format(class_loss.item()))
            print('Train Reconstruction loss: {} %'.format(recon_loss.item()))
            class_loss_hist.append(class_loss.item())
            recon_loss_hist.append(recon_loss.item())
            D_loss_hist.append(D_loss.item())
            D_loss_cat_hist.append(D_loss_cat.item())
            D_loss_gauss_hist.append(D_loss_gauss.item())
            G_loss_hist.append(G_loss.item())
            
            if test_F1:  
                F1_train,F1_val=get_f1_scores(Q,[.6])
                F1_train_hist.append(F1_train)
                F1_val_hist.append(F1_val)
                
            
    end = time.time()
    print('Training time: {} seconds'.format(end - start))
    return Q, P,[class_loss_hist,recon_loss_hist,D_loss_hist,D_loss_cat_hist,D_loss_gauss_hist,G_loss_hist,F1_train_hist,F1_val_hist]




def save_model(filename,save_path,p,q):
    
    state = {'state_dict': p.state_dict(),
             'optimizer': p.state_dict() }
    torch.save(state, save_path+ 'P'+ filename)
    
    state = {'state_dict': q.state_dict(),
             'optimizer': q.state_dict() }
    torch.save(state, save_path+ 'Q'+filename)

    return


def get_outputs(model,net_name,directory,data):
    
    
    all_outputs=np.zeros((len(data),1000))
    
    model.eval()
    
    if 'training' in directory:
        idx_output=idx_to_name
    if 'validation' in directory:
        idx_output=idx_to_name_v
    
        
    for t in range(len(all_outputs)):
            
            print(str(t)+'/'+str(len(data)))
        
            fullname = directory+idx_output[t]+'.jpg'
    
            image = Image.open(fullname).convert('RGB')
    
            image = preprocessing(image)
            
    
            image = Variable(image.cuda()).view((1,3,64,64))

            output=model(image)[0]
            output_np=output.cpu().detach().numpy()
            

            
            all_outputs[t]=output_np
            
    np.save(net_name+'_all_outputs',all_outputs)
    
    return all_outputs

def create_text_files(thresholds, max_per_sample,net_name,array_of_outputs):
    
    
    if 'train' in net_name:
        idx_output=idx_to_name
    if 'val' in net_name:
        idx_output=idx_to_name_v
    

    filenames=[net_name+'_t'+ str(thresholds[i])+'.txt' for i in range(len(thresholds))]
    
    over_100=[]
    
    for j,threshold in enumerate(thresholds):
        
        
        
        
        array_of_outputs_sorted=-np.sort(-array_of_outputs,1)[:,:max_per_sample]
        array_of_outputs_argsorted=np.argsort(-array_of_outputs,1)[:,:max_per_sample]
        
        
        binary=array_of_outputs_sorted>threshold
        
        
        f=open(filenames[j], 'w')
        over_100_true=False
        
        for t in range(len(array_of_outputs)):
            

            
            if t%500==0:
                pass;
            
            indices=np.argwhere(binary[t]==1)
            indices.shape=len(indices)
            
            if len(indices)>100:
                over_100_true=True
                over_100.append(t)
            
            line=idx_output[t]+'\t'
            for i in indices:
                line+=idx_to_umls[array_of_outputs_argsorted[t,i]]+','
                
            line=line[:-1]
            line+='\n'
            f.write(line)
        f.close()
        
        if over_100_true==True:
            print('there is at least one sample with over 100 UMLS codes')
    return filenames


def get_f1_scores(model,thresholds):

    train_outputs=get_outputs(model,'q_customtrain',training_path,training_data)
    val_outputs=get_outputs(model,'q_customval',validation_path,validation_data)
    

    filenames_train=create_text_files(thresholds,25,'train',train_outputs)
    filenames_val=create_text_files(thresholds,25,'val',val_outputs)

    from Evaluation.evaluate_f1 import main as F1
    
    F1_score_train=[]
    F1_score_val=[]
    for t_file,v_file in zip(filenames_train,filenames_val):
        
        F1_score_train.append(F1(t_file,'ImageCLEFmedCaption2019_Training-Concepts_comma.txt'))
        F1_score_val.append(F1(v_file,'ImageCLEFmedCaption2019_Validation-Concepts_comma.txt'))

    return F1_score_train,F1_score_val







if __name__ == '__main__':    
    
    train_labeled_loader, train_unlabeled_loader, valid_loader = get_loaders(training_data,validation_data,validation_data,training_path,validation_path,validation_path)
    
    
    print('dataloaders created')

    tic=time.time()
    print(tic)
    Q, P, hists= generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader)
    toc=time.time()
    print(toc)