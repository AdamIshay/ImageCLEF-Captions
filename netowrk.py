# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:07:28 2019

@author: Adam
"""

import os
os.chdir('/home/adam/Python_stuff/imageclef')

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir, makedirs, getcwd, remove
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from torch.optim import lr_scheduler
from torch.autograd import Variable

from preprocessing_labels import training_data,validation_data,idx_to_name,idx_to_umls,umls_to_idx

from preprocessing_images import Data,Data_save,preprocessing



training_ground_truth='/home/adam/Python_stuff/IMAGEclef/one_epoch_t3'
training_ground_truth='/home/adam/Python_stuff/IMAGEclef/ImageCLEFmedCaption2019_Training-Concepts'



def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):
    use_gpu = torch.cuda.is_available()

    dataset_sizes = {'train': len(dataloaders['train'].dataset), 
                     'valid': len(dataloaders['valid'].dataset)}

    for epoch in range(num_epochs):
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                #breakpoint()
                model.train(False)

            running_loss = 0.0
            batch_count=0;
            for inputs, labels in dataloaders[phase]:
                
                batch_count+=1;
                #breakpoint()
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.float().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels.float())

                optimizer.zero_grad()

                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                #running_corrects += torch.sum(preds == labels.data)
                #breakpoint()
                print('epoch: ' +str(epoch+1)+'/'+str(num_epochs)+'    batch: ' + 
                      str(batch_count)+'/'+str(dataloaders[phase].__len__())+
                      '\t Loss: ' + str(loss.detach().item()));
                
                
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                

        print('Epoch [{}/{}] train loss: {:.4f} ' 
              'valid loss: {:.4f}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, 
                valid_epoch_loss))

    return model




def get_outputs(model,net_name,directory,data):
    '''saves and returns the outputs from the model after feeding in the dataset'''
    
    all_outputs=np.zeros((len(data),1000))
    
    model.eval()
    
    for t in range(len(all_outputs)):
            
            print(str(t)+'/'+str(len(data)))
        
            fullname = directory+idx_to_name[t]+'.jpg'
    
            image = Image.open(fullname).convert('RGB')
    
            image = preprocessing(image)
            
    
            image = Variable(image.cuda()).view((1,3,224,224))
            
            output=model(image.view(1,3,224,224))[0]
            
            output_np=output.cpu().detach().numpy()
            
            #indices_threshold=np.argwhere(output_np>threshold)
            
            all_outputs[t]=output_np
            
    np.save(net_name+'_all_outputs',all_outputs)
    
    return all_outputs

def create_text_files(thresholds, max_per_sample,net_name,array_of_outputs):
    '''creates the formatted submission text files for each threshold
    thresholds: list of thresholds on the interval [0,1]
    max_per_sample: the maximum amount of concepts allowed per sample
    '''
    
    filenames=[net_name+'_t'+ str(thresholds[i]) for i in range(len(thresholds))]
    
    over_100=[]
    
    for j,threshold in enumerate(thresholds):

        array_of_outputs_sorted=-np.sort(-array_of_outputs,1)[:,:max_per_sample]
        array_of_outputs_argsorted=np.argsort(-array_of_outputs,1)[:,:max_per_sample]
        
        
        binary=array_of_outputs_sorted>threshold
        
        
        #check that sorted and argsorted indices match up
        
        
        f=open(filenames[j], 'w')
        over_100_true=False
        
        for t in range(len(array_of_outputs)):
            

            
            #indices_threshold=np.argwhere(array_of_outputs>threshold)
            if t%500==0:
                pass;#print('t is ' +str(t))
            
            indices=np.argwhere(binary[t]==1)
            indices.shape=len(indices)
            
            if len(indices)>100:
                over_100_true=True
                over_100.append(t)
            
            line=idx_to_name[t]+'\t'
            for i in indices:
                line+=idx_to_umls[array_of_outputs_argsorted[t,i]]+','
                
            line=line[:-1]
            line+='\n'
            f.write(line)
        f.close()
        
        if over_100_true==True:
            print('there is at least one sample with over 100 UMLS codes')
    return


def everything(model,net_name,data,data_path, thresholds, max_per_sample, all_outputs=None): 
    
    if all_outputs is None: #gets outputs if all_outputs is None
        all_outputs=get_outputs(model,net_name, data_path,data) #np.array of outputs
        
    
    create_text_files(thresholds,max_per_sample, net_name,all_outputs)
    
    




def load_model(model_path,resnet):
    
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        resnet.load_state_dict(checkpoint['state_dict'])
        resnet.load_state_dict(checkpoint['optimizer'])
        
        
    
        if use_gpu:
            resnet = resnet.cuda()
        
        
        
        print("=> loaded checkpoint '{}' "
                  .format(model_path))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    return resnet



def save_model(filename,net):

    state = {'state_dict': net.state_dict(),
             'optimizer': net.state_dict() }
    torch.save(state, '/home/adam/Python_stuff/imageclef/models/'+filename)



if __name__== '__man__':
    
    
    
    
    #train neural network
    #produce outputs for each and save
    #threshold them, and find F1 scores with script
    
    
    
    
    #use_gpu = torch.cuda.is_available()
    use_gpu = True
    
    
    training_path='/home/adam/Python_stuff/imageclef/training-set/'
    validation_path='/home/adam/Python_stuff/imageclef/validation-set/'
    
    model_path=None
    #model_path = 'C:/Users/Adam/Python_stuff/IMAGEclef/Models/resnet50_P001_mlsml4epochs'
    #all_outputs= np.load('7_epochs_all_outputs.npy')
#    all_outputs=outputs
    
    
    #resnet = models.resnet50(pretrained=True)
    
    from resnet_new_class import resnet50
    resnet = resnet50(pretrained=True)
    
    
    
    if model_path is None:
        
        
        
        training_ds=Data(training_data,directory=training_path,transform=preprocessing)
        
        training_dl=DataLoader(training_ds,batch_size=64,shuffle=True, num_workers=0)
        
        
        
        #validation_ds=Data(training_data, directory=validation_path,transform=preprocessing)
        
        validation_ds=Data(validation_data, directory=validation_path,transform=preprocessing)
        
        validation_dl=DataLoader(validation_ds,batch_size=32,shuffle=True, num_workers=0)
        
        
        
        
        
        
# =============================================================================
#         #previous freezing method
#         child_counter = 0
#         for child in resnet.children():
#             if True:
#                 print("child ",child_counter," was frozen")
#                 param_counter=0
#                 child_counter+=1
#                 print('childcounter is ' +str(child_counter))
#                 for param in child.parameters():
#                     param_counter+=1
#                     print('param_counter is' + str(param_counter))
#                     param.requires_grad = False
# =============================================================================
    
# =============================================================================
# keep all unfrozen
#     ct = 0
#     for child in resnet.children():
#         ct += 1
#         if ct < 8:
#             for param in child.parameters():
#                 param.requires_grad = False
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     
#         resnet.fc.weight.requires_grad=True
#     
# =============================================================================
    
        #breakpoint()
        #inputs, labels = next(iter(training_dl))
        
        if use_gpu:
            resnet = resnet.cuda()
            #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())   
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        #outputs = resnet(inputs)
        #outputs.size()
        
        
        #criterion = torch.nn.BCEWithLogitsLoss()
        criterion = torch.nn.BCELoss()
        #criterion = torch.nn.MultiLabelSoftMarginLoss()
        optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
        dloaders = {'train':training_dl, 'valid':validation_dl}
        
        
        tic=time.time()
        
        model = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=40)
    
        toc=time.time()
        
        
        print('time is' +str(toc-tic))
        #net_name='P001'
        
        #save_model(net_name,model)
        
        
        
        
    else:
        model=load_model(model_path,resnet)
        
        
    thresholds=[.5,1,2]
    
    
    #UMLS code limit per sample (100 maximum)
    
    
    
    
    
    #everything(model,'7_epoch',training_data,training_path,thresholds,25,all_outputs=all_outputs)
    


#4,12,

































    # =============================================================================
    # child_counter = 0
    # for child in resnet.children():
    #     if child_counter < 7:
    #         print("child ",child_counter," was frozen")
    #         for param in child.parameters():
    #             param.requires_grad = False
    #     elif child_counter == 7:
    #         children_of_child_counter = 0
    #         for children_of_child in child.children():
    #             if children_of_child_counter < 0:
    #                 for param in children_of_child.parameters():
    #                     param.requires_grad = False
    #                 print('child ', children_of_child_counter, 'of child',child_counter,' was frozen')
    #             else:
    #                 print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
    #             children_of_child_counter += 1
    # 
    #     else:
    #         print("child ",child_counter," was not frozenn")
    #     child_counter += 1
    # =============================================================================
    