# FaceNet Implementation on DogFace Dataset
Paper To Code implementation of Facenet on dog-face dataset using custom online Hard-Triplet mining


Original Paper- [ArxViv](https://arxiv.org/abs/1503.03832)

This custom implementation of FaceNet trained on dog face dataset. My approach was to read the paper (FaceNet: A Unified Embedding for Face Recognition and Clustering) and try to implement the model from my interpretation of the paper. I have used pytorch for the implementation.



## Model Architecture and Training Design

![Pic of Model](https://github.com/kvsnoufal/Pytorch-FaceNet-DogDataset/blob/master/doc/inception.png)

The objective of the model is to generate embeddings that satisfy this these 2 constraints:
Same faces are close to each other in embedding space
Different faces are far away

The loss function does exactly this.
A training step would comprise the following:
Select 3 images

Anchor image (a)- image of a person A
Positive sample (p)-another image of person A
Negative sample (n) - image of person B

2. Train the model to minimize the triplet loss:
![Triplet Loss function](https://github.com/kvsnoufal/Pytorch-FaceNet-DogDataset/blob/master/doc/tripletloss.gif)

![training](https://github.com/kvsnoufal/Pytorch-FaceNet-DogDataset/blob/master/doc/trainging.gif)



One of the optimizations to the training processes proposed in the paper is the triplet selection process - Hard Triplet Mining. In order to reduce the time taken for convergence of the model, triplets which can contribute to model improvement need to be carefully selected. 
So for an anchor image, we select a positive image that has embedding farthest from anchor's - Hard Positive. And we select a negative image that has embedding closest to the anchor's - Hard Negative.

![Hard Triplet Mining](https://github.com/kvsnoufal/Pytorch-FaceNet-DogDataset/blob/master/doc/hardnegative.png)


The training process is essentially, the neural network learning to generate embeddings that minimizes the triplet loss. This ensures the trained model would embed images of the same person very close to each other.
## Building a dog search engine
I trained facenet on dog dataset using a custom dataloader that implements hard triplet mining.

![Tensorboard](https://github.com/kvsnoufal/Pytorch-FaceNet-DogDataset/blob/master/doc/loss.png)

![Tensorboard Embedding](https://github.com/kvsnoufal/Pytorch-FaceNet-DogDataset/blob/master/doc/embedding.gif)


![Web gif](https://github.com/kvsnoufal/Pytorch-FaceNet-DogDataset/blob/master/doc/facenet_webinterface.gif)

Check out youtube video :
[Youtube Link](https://youtu.be/0VZiECk8NjM)












