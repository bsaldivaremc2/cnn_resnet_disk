
# Convolutional Neural Network including residual network and train images from disk

I made a library (**under construction**) that allows easy construction of convolutional neural networks and includes residual networks blocks as stated in [1]. In addition you can train images without loading them all into memory. Only the images that are being trained are those loaded into memory. Therefore making it easy and efficient to train when it is not possible to load them all because of computational constrains.  

I present an example of use for landmarks regression. Using the VGG annotator output format [2].



**References**  
[1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” ArXiv151203385 Cs, Dec. 2015

[2] @misc{ dutta2016via, author = "Dutta, A. and Gupta, A. and Zissermann, A.", title = "{VGG} Image Annotator ({VIA})", year = "2016",
howpublished = "http://www.robots.ox.ac.uk/~vgg/software/via/",
note = "Accessed: 2018/03/26" }


```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import imp
import os
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle

libraries_dir = 'YourFolderForThisLibrary/'
import sys
sys.path.insert(0, libraries_dir)

import cnn_model_resnet
imp.reload(cnn_model_resnet)
from cnn_model_resnet import *

import dl_utils
imp.reload(dl_utils)

dfx = 'dafaframe_with_landmarks_and_absolute_path_to_images.csv'

data= '/home/your_user/path_to_your_data/'

#Location of your Images
imgs_dir= data+'Images/'

#Location of your CSVs
csvs = data+'CSVs/'
dfx = csvs+dfx

dfx = pd.read_csv(dfx)
print(dfx.shape)

print(dfx.columns)
#The previous line outputs:
## x0,x1,x2...x10,y0,y1,y2,...y10,filename, fullname

#The fullname field in my dataframe has the name of the images files but with an absolute path to other computer
##Therefore I am extracting only the filename without folder and adding the image folder where it is
dfx['fullpath']=imgs_dir+dfx['fullname'].apply(lambda x: x.split("/")[-1])

#I have 11 landmarks so I am creating a label for each one
max_point = 11
x_labels=['x'+str(_) for _ in range(0,max_point)]
y_labels=['y'+str(_) for _ in range(0,max_point)]

#Here I am testing that I can load just a batch of images into memory.
##x_label is the column label for the absolote path to the image
##xp_label and yp_label are the columns for the landmark points for x and y coordinates.
## batch_size: defines how many rows to load
## offset: defines from which starting point load the mini-batch
## resize_wh: will resize the images to this size, mandatory field.
## toGray: will turn the images into grayscale if set to True
# Outputs
##imgs: is the x matrix of images of four dimensions [batch_size,image_width,image_height,channels]
## fxs and fys : are the landmarks for x and y coordinates in a numpy array, join them across the 1 axis
## to create a target y 
imgs, fxs,fys = df_x_y (dfx,x_label='fullpath',xp_label=x_labels,yp_label=y_labels,
              batch_size=5,offset=10,resize_wh=(64,92),toGray=False)


# Here I am separating the training and testing, 80% training.
rows = dfx.shape[0]
train_rows = int(round(rows*0.8))
test_rows = rows - train_rows
df_train = dfx.head(train_rows)
df_test = dfx.tail(test_rows)
```

# Simple residual layer block

We need three main components, the model, the parameters to load from disk and the training function.  
For the model just add one of those lists inside the model list to include in the model.  
The dfxy_args are the parameters described previously so the training could load each image just when the training on these images is required instead of loading them all into memory.   
the train function lr: learning rate, iters: number of iterations, the save_name is to save the model state into a file that later could be used for prediction or testing. The opt_mode is for regression or classification. For landmarks is regression.


```python
model = [
    ['conv',{'filter_size':3,'layer_depth':2}],
    ['bn'],
    ['relu'],
    ['max_pool',{'kernel':[1,2,2,1],'strides':[1,1,1,1],'padding':'SAME'}],
    ['drop_out',{'prop':0.9}],
    ['res_131',{'depths':[4,4,8]}]
    
]
dfxy_args={'df':dfx,'x_label':'fullpath','xp_label':x_labels,'yp_label':y_labels,
              'batch_size':8,'offset':10,'resize_wh':(64,92),'toGray':False}

train(dfxy_args,model=model,iters=1,lr=0.0001,
          save_model=True,save_name='test_model',
          restore_model=False,restore_name='test_model',
          v=True,opt_mode='regression')
```

# 50 layer


```python
model = [
    ['conv',{'filter_size':7,'layer_depth':64,'strides':[1,2,2,1]}],
    ['bn'],
    ['relu'],
    ['max_pool',{'kernel':[1,3,3,1],'strides':[1,2,2,1],'padding':'SAME'}],   
]
b2 = [ ['res_131',{'depths':[64,64,256]}] for _ in range(0,3) ]
b3 = [ ['res_131',{'depths':[128,128,512]}] for _ in range(0,4) ]
b4 = [ ['res_131',{'depths':[256,256,1024]}] for _ in range(0,6) ]
b5 = [ ['res_131',{'depths':[512,512,2048]}] for _ in range(0,3) ]
model.extend(b2)
model.extend(b3)
model.extend(b4)
model.extend(b5)

dfxy_args={'df':df_train,'x_label':'fullpath','xp_label':x_labels,'yp_label':y_labels,
              'batch_size':8,'offset':10,'resize_wh':(64,92),'toGray':False}

train(dfxy_args,model=model,iters=1,lr=0.0001,
          save_model=True,save_name='test_model',
          restore_model=False,restore_name='test_model',
          v=True,opt_mode='regression')
```

# 101 layer


```python
model = [
    ['conv',{'filter_size':7,'layer_depth':64,'strides':[1,2,2,1]}],
    ['bn'],
    ['relu'],
    ['max_pool',{'kernel':[1,3,3,1],'strides':[1,2,2,1],'padding':'SAME'}],   
]
b2 = [ ['res_131',{'depths':[64,64,256]}] for _ in range(0,3) ]
b3 = [ ['res_131',{'depths':[128,128,512]}] for _ in range(0,4) ]
b4 = [ ['res_131',{'depths':[256,256,1024]}] for _ in range(0,23) ]
b5 = [ ['res_131',{'depths':[512,512,2048]}] for _ in range(0,3) ]
model.extend(b2)
model.extend(b3)
model.extend(b4)
model.extend(b5)

dfxy_args={'df':dfx,'x_label':'fullpath','xp_label':x_labels,'yp_label':y_labels,
              'batch_size':8,'offset':10,'resize_wh':(64,92),'toGray':False}

train(dfxy_args,model=model,iters=1,lr=0.0001,
          save_model=True,save_name='test_model',
          restore_model=False,restore_name='test_model',
          v=True,opt_mode='regression')
```
