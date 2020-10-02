import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import config
from train import model




class DogDatasetNaive(Dataset):
    def __init__(self,paths,labels):
        self.paths=paths
        self.labels=labels

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,idx):
        anchor_file=self.paths[idx]
        anchor_label=self.labels[idx]

        positive_idx=np.argwhere((self.labels==anchor_label)&(self.paths!=anchor_file))
        positives=self.paths[positive_idx].flatten()
        positive=np.random.choice(positives)

        negatives_idx=np.argwhere(self.labels!=anchor_label)
        negatives=self.paths[negatives_idx].flatten()
        negative=np.random.choice(negatives)
        anchors=np.array(Image.open(anchor_file))
        positives=np.array(Image.open(positive))
        negatives=np.array(Image.open(negative))

        anchors=np.transpose(anchors,(2,0,1)) /255.0
        positives=np.transpose(positives,(2,0,1)) /255.0
        negatives=np.transpose(negatives,(2,0,1))  /255.0



        return {"anchor":torch.tensor(anchors,dtype=torch.float,device=torch.device(config.DEVICE)),\
                "positive":torch.tensor(positives,dtype=torch.float,device=torch.device(config.DEVICE)),\
                "negative":torch.tensor(negatives,dtype=torch.float,device=torch.device(config.DEVICE))}


def get_tensors(ps):
    ims=[np.array(Image.open(p)) for p in ps]
    ims=[np.transpose(p,(2,0,1))/255.0 for p in ims]
    ims=np.array(ims)
    return torch.tensor(ims,dtype=torch.float,device=torch.device(config.DEVICE))
class embedDataset(Dataset):
    def __init__(self,paths):
        self.paths=paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self,idx):
        file=self.paths[idx]
        img=get_tensors([file]).squeeze(0)
        return {"image":img}

class DogDatasetHard(Dataset):
    def __init__(self,paths,labels):
        self.paths=paths
        self.labels=labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        
        with torch.no_grad():
            if np.random.random() < config.SAMPLING_RATIO:
                anchor_file=self.paths[idx]
                anchor_label=self.labels[idx]

                positive_idx=np.argwhere((self.labels==anchor_label)&(self.paths!=anchor_file))
                positives=self.paths[positive_idx].flatten()

                positives_model_input=get_tensors(positives)
                positives_embeddings=model(positives_model_input).detach()

                
                anchor_model_input=get_tensors([anchor_file])
                anchor_embedding=model(anchor_model_input)

                distaps=F.pairwise_distance(anchor_embedding.repeat(len(positives_embeddings),1),\
                positives_embeddings,\
                    2 )

                harderst_p_index=torch.argmax(distaps)
                hardest_positive=positives_model_input[harderst_p_index]

                negatives_idx=np.argwhere(self.labels!=anchor_label)
                negatives=self.paths[negatives_idx].flatten()
                negatives=np.random.choice(negatives,100,replace=False)
                negatives_model_input=get_tensors(negatives)

                negatives_embeddings=model(negatives_model_input)
                dist_nps=F.pairwise_distance(anchor_embedding.repeat(len(negatives_embeddings),1),\
                negatives_embeddings,\
                    2 )

                harderst_n_index=torch.argmin(dist_nps)
                hardest_negative=negatives_model_input[harderst_n_index]

                return {"anchor":anchor_model_input.squeeze(0).to(config.DEVICE),
                        "positive": hardest_positive.to(config.DEVICE),
                        "negative":hardest_negative.to(config.DEVICE)}
            else:

                anchor_file=self.paths[idx]
                anchor_label=self.labels[idx]

                positive_idx=np.argwhere((self.labels==anchor_label)&(self.paths!=anchor_file))
                positives=self.paths[positive_idx].flatten()
                positive=np.random.choice(positives)

                negatives_idx=np.argwhere(self.labels!=anchor_label)
                negatives=self.paths[negatives_idx].flatten()
                negative=np.random.choice(negatives)
                anchors=np.array(Image.open(anchor_file))
                positives=np.array(Image.open(positive))
                negatives=np.array(Image.open(negative))

                anchors=np.transpose(anchors,(2,0,1)) /255.0
                positives=np.transpose(positives,(2,0,1)) /255.0
                negatives=np.transpose(negatives,(2,0,1))  /255.0



                return {"anchor":torch.tensor(anchors,dtype=torch.float,device=torch.device(config.DEVICE)),\
                        "positive":torch.tensor(positives,dtype=torch.float,device=torch.device(config.DEVICE)),\
                        "negative":torch.tensor(negatives,dtype=torch.float,device=torch.device(config.DEVICE))}                