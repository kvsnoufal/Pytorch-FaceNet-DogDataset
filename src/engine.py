import torch
import torch.nn as nn
import config
from tqdm import tqdm
import numpy as np
import os
from train import model
class Engine:
    def __init__(self,criterion,optimizer,device):
        self.criterion=criterion
        self.optimizer=optimizer
        self.device=device
        

    def train(self,dataloader,epoch):
        
        model.train()
        losses=0
        for batch in tqdm(dataloader,total=len(dataloader)):
            model.zero_grad()
            anchor_embedding=model(batch["anchor"])
            positive_embedding=model(batch["positive"])
            negative_embedding=model(batch["negative"])
            loss=self.criterion(anchor_embedding,positive_embedding,negative_embedding)
            loss.backward()
            self.optimizer.step()
            losses+=loss.item()
        epoch_loss=losses/len(dataloader)
        return epoch_loss
    def evaluate(self,dataloader,epoch):
        
        model.eval()
        val_loss_value=0
        with torch.no_grad():
            for batch in dataloader:
                anchor_embedding=model(batch["anchor"])
                positive_embedding=model(batch["positive"])
                negative_embedding=model(batch["negative"])
                loss=self.criterion(anchor_embedding,positive_embedding,negative_embedding)
                val_loss_value+=loss.item()
            val_epoch_loss=val_loss_value/len(dataloader)
        return val_epoch_loss
    def summarize_embedding(self,dataloader,writer,epoch):
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                embeddings=model(batch["image"])
                writer.add_embedding(embeddings,global_step=epoch,label_img=batch["image"])
        