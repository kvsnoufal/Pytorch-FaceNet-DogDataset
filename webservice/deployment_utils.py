import sys
sys.path.insert(0,"../")
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

PRETRAINED_MODEL="../modelsave/best_model.pth"
ARTEFACTS="../artefacts"
def load_model():
    model=GoogLeNet()
    model.load_state_dict(torch.load(PRETRAINED_MODEL))
    return model
def load_artefacts():
    knn=joblib.load(os.path.join(ARTEFACTS,"knn.pkl"))
    le=joblib.load(os.path.join(ARTEFACTS,"lEncoder.pkl"))
    return knn,le

    

