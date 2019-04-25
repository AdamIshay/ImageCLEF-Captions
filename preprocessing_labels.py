# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:42:58 2019

@author: Adam
"""
import csv
import os
import numpy as np
import re
import matplotlib.pyplot as plt



# get the list of UMLS codes

#open string concepts file
with open('ImageCLEFmedCaption2019_String-Concepts.csv', 'r', encoding='utf8') as f:
  reader = csv.reader(f)
  UMLS_concepts = list(reader)
  

idx_to_umls={}
umls_to_idx={}
concepts=[]
for i,x in enumerate(UMLS_concepts):
    
    
    c1,c2=re.split(r'\t+',x[0])
    umls_to_idx[c1]=i
    idx_to_umls[i]=c1
    
    concepts.append(c2)

#load training data ----------------------------------------------

#open training concepts file
with open('ImageCLEFmedCaption2019_Training-Concepts.txt', 'r', encoding='utf8') as f:
  reader = csv.reader(f)
  training = list(reader)


idx_to_name={}
name_to_idx={}

training_data={}

image_id=[];image_concepts=[]

for i,t in enumerate(training):
    
    split=re.split(r'\t+|;',t[0])
    image_id.append(split[0])
    
    sample_name=split[0]
    name_to_idx[sample_name]=i
    idx_to_name[i]=sample_name
    
    image_concept=split[1:]

    
    image_concept=[umls_to_idx[j] for j in image_concept]
    
    training_data[i]=image_concept
    image_concepts.append(image_concept)


all_concepts=[]

for img_concept in image_concepts:
    all_concepts=all_concepts+img_concept



all_concepts_np=np.array(all_concepts)
image_concepts_np=np.array(image_concepts)



num_of_concepts=[len(x) for x in image_concepts]

num_of_concepts_np=np.array(num_of_concepts)



a,b=np.unique(all_concepts_np,return_counts=True)


indices_ordered=a[np.argsort(b)[::-1]]

concepts_ordered=[[concepts[x],1*(all_concepts_np==x).sum()] for x in indices_ordered]

top_1000_indices=indices_ordered[:1000]
top_1000_concepts=[m[0] for m in concepts_ordered[:1000]]



#load validation data ----------------------------------------------------

validation_data={}


#open validation concepts file 
with open('ImageCLEFmedCaption2019_Validation-Concepts.txt', 'r', encoding='utf8') as f:
  reader = csv.reader(f)
  validation = list(reader)

idx_to_name_v={}
name_to_idx_v={}

image_id_v=[];image_concepts_v=[]

for i,t in enumerate(validation):
    
    split=re.split(r'\t+|;',t[0])
    image_id_v.append(split[0])
    
    sample_name=split[0]
    name_to_idx_v[sample_name]=i
    idx_to_name_v[i]=sample_name
    
    image_concept_v=split[1:]
    
    image_concept_v=[umls_to_idx[j] for j in image_concept_v]
    
    validation_data[i]=image_concept_v
    image_concepts_v.append(image_concept_v)

all_concepts_v=[]

for img_concept_v in image_concepts_v:
    all_concepts_v=all_concepts_v+img_concept_v



all_concepts_v_np=np.array(all_concepts_v)
image_concepts_v_np=np.array(image_concepts_v)



num_of_concepts_v=[len(x) for x in image_concepts_v]

num_of_concepts_v_np=np.array(num_of_concepts_v)


if __name__ =='__main__':
    
    with open('ImageCLEFmedCaption2019_Training-Concepts.txt', 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        training = list(reader)


    image_id=[];image_concepts=[]
    
    for t in training:
        
        split=re.split(r'\t+|;',t[0])
        image_id.append(split[0])
        #breakpoint()
        image_concept=split[1:]
        
        image_concept=[code_to_idx[j] for j in image_concept]
        
        image_concepts.append(image_concept)
    
    all_concepts=[]
    
    for img_concept in image_concepts:
        all_concepts=all_concepts+img_concept
    
    
    
    all_concepts_np=np.array(all_concepts)
    image_concepts_np=np.array(image_concepts)
    
    
    
    num_of_concepts=[len(x) for x in image_concepts]
    
    num_of_concepts_np=np.array(num_of_concepts)
    
    
    
    
    