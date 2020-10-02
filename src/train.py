import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from models import GoogLeNet
import loss_fn
import config
import engine

##global definitions##
global model
model=GoogLeNet().to(config.DEVICE)
###############



from dataset import DogDatasetNaive,get_tensors,embedDataset,DogDatasetHard




def load_df():
    paths=glob(config.IMG_PATH)
    labels=[path.split("\\")[-1][:-4].split(".")[0] for path in paths]
    file_name=[path.split("\\")[-1][:-4].split(".")[1] for path in paths]
    df=pd.DataFrame({"img_paths":paths,"labels":labels,"file_name":file_name})
    df["img_paths"]=df["img_paths"].apply(lambda x: x.replace("\\","/"))
    return df#.head(500)




if __name__ == "__main__":
    os.makedirs(config.MODEL_SAVEPATH,exist_ok=True)
    os.makedirs(config.LOG_DIR,exist_ok=True)
    writer = SummaryWriter(config.LOG_DIR)
    
    df=load_df()
    val_labels=np.random.choice(df["labels"].unique(),int(0.3*df["labels"].nunique()))
    val_df=df[df["labels"].isin(val_labels)]
    df=df.loc[df.labels.isin(val_df.labels)==False]
    df=df.reset_index(drop=True)
    val_df=val_df.reset_index(drop=True)
    inf_df=glob(config.EMBED_IMG_PATH)
    inf_df=[t.replace("\\","/") for t in inf_df]

    print("TRAIN_DATASET : {} VALIDATION_DATASET : {} EMBEDDING_LOG_DATASET: {}".format(len(df),len(val_df),len(inf_df)))

    dogdata=DogDatasetHard(df["img_paths"].values,df.labels.values)
    val_data=DogDatasetHard(val_df["img_paths"].values,val_df.labels.values)
    inf_data=embedDataset(inf_df)

    dataloader = DataLoader(dogdata, batch_size=config.BATCH_SIZE,
                        shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=0)
    inf_dataloader = DataLoader(inf_data,batch_size=57,
                        shuffle=False, num_workers=0 )  

    
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    criterion=loss_fn.TripletLoss().to(config.DEVICE)
    E=engine.Engine(criterion=criterion,optimizer=optimizer,device=config.DEVICE)

    best_loss=999
    losses=[]       
    for epoch in range(config.EPOCHS):
        train_loss=E.train(dataloader,epoch)
        print(f"TRAINING STEP: {epoch} LOSS: {train_loss}")

        val_loss=E.evaluate(val_dataloader,epoch)
        print(f"TRAINING STEP: {epoch} LOSS: {val_loss}")


        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)


        if val_loss<best_loss:
            best_loss=val_loss
            print("Saving Best Model ",epoch)
            torch.save(model.state_dict(),os.path.join(config.MODEL_SAVEPATH,"best_model.pth"))
    writer.flush()
    writer.close()