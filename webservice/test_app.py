from flask import render_template
# from app import app
from flask import Flask
import os
from glob import glob
import json
import numpy as np
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import json
import deployment_utils
from PIL import Image
import torch


knn,le=deployment_utils.load_artefacts()
model=deployment_utils.load_model()
model.cuda()
model.eval()

def query_image():
    query_image=np.transpose(np.array(Image.open("./image.jpg")),(2,0,1))/255.0
    q_input=torch.tensor(query_image,dtype=torch.float).unsqueeze(0)
    q_embedding=model(q_input.cuda()).detach().cpu().numpy()
    ranked=np.argsort(knn.predict_proba(q_embedding)[0])[::-1]
    ranked=list(le.inverse_transform(ranked)[:5])
    ranked=[t for t in ranked if t!='999']
    return ranked[:4]






app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    
    return render_template('index.html', title='Home',ims={})


@app.route('/search', methods=['GET','POST'])
def search_func():
    clicked=None
    if request.method == "POST":
        print("post hello")
        # print("here" ,request.get_json(force=True))
        print(request.data)
        print(request.form)
        print(request.files)
        # print(dir(request))

        # clicked=request.json['data']
        file = request.files['img']
        file.save("./image.jpg")
        print(type(file))
    print("test called")

    img_index=["1","2","3","4"]
    img_index=query_image()
    results={}
    for i,img in enumerate(img_index):
        files=glob(f"./static/imgs/{img}.*")
        images=["/static/imgs/"+t.split("\\")[-1] for t in files]
        np.random.shuffle(images)
        results[f"results{i+1}"]=images
        
    print(results)
    # return render_template('index.html', title='Home',ims=results)
    return results
