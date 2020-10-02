import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from model import GoogLeNet

from loss_fn import TripletLoss
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib

PRETRAINED_MODEL="./modelsave/best_model.pth"
INDEXING_PATH="../input/indexing/*.jpg"
OUTPUT_PATH="artefacts"
def load_model():
    model=GoogLeNet()
    model.load_state_dict(torch.load(PRETRAINED_MODEL))
    return model

class DogIndexingDataset(Dataset):
    def __init__(self,paths,labels):
        self.paths=paths
        self.labels=labels

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,idx):
        anchor_file=self.paths[idx]
        anchor_label=self.labels[idx]

       
        anchors=np.array(Image.open(anchor_file))
        
        anchors=np.transpose(anchors,(2,0,1)) /255.0
        
        return {"images":torch.tensor(anchors,dtype=torch.float),
                "label":anchor_label}

def gen_index_labels():
    indexed_files=glob(INDEXING_PATH)
    label_index=[t.split("\\")[-1].split(".")[0] for t in indexed_files]
    le=LabelEncoder()
    label_=le.fit_transform(label_index)
    return indexed_files,label_index,le,label_

if __name__ == "__main__":
    model=load_model()
    model.cuda()
    model.eval()
    indexed_files,label_index,le,label_=gen_index_labels()

    ds=DogIndexingDataset(indexed_files,label_index)
    dataloader = DataLoader(ds, batch_size=2,
                            shuffle=False, num_workers=0)
    embeddings=[]
    labels=[]
    for bs in dataloader:
        labels.extend(bs["label"])
        embeddings.extend(model(bs["images"].cuda()).detach().cpu().numpy())
    knn=KNeighborsClassifier(3)
    knn.fit(np.array(embeddings),np.array(label_)) 

    os.makedirs(OUTPUT_PATH,exist_ok=True)
    joblib.dump(knn,os.path.join(OUTPUT_PATH,"knn.pkl"))
    joblib.dump(le,os.path.join(OUTPUT_PATH,"lEncoder.pkl"))
    
