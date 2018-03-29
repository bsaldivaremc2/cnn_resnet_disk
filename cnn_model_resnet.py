import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imp
import os
import sys
from dl_utils import *
from PIL import Image


def train(df_xy_args,model=None,iters=10,lr=0.001,
          save_model=True,save_name=None,
          restore_model=False,restore_name=None,
          v=False,opt_mode='classification'):
    
    idf = df_xy_args['df']   
    is_gray = df_xy_args['toGray']
    if is_gray==True:
        in_channels=1
    else:
        in_channels=3
    
    xs0 = df_xy_args['resize_wh']
    ixs = [None,xs0[1],xs0[0],in_channels]
    iys = [None,idf[df_xy_args['xp_label']].shape[1]*2]
    

    #ixs,iys=ix.shape,iy.shape
    
    xi,y_,learning_rate,train_bool,loss,train_step,stats = create_model(ixs,iys,model=model,opt_mode=opt_mode)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as s:
        if restore_model==True:
            if restore_name==None:
                print("No model file specified")
                return
            else:
                saver.restore(s,restore_name)
        else:
            s.run(init_op)
        fd={learning_rate:lr,train_bool:True}
        for _ in range(0,iters):
            
            idfs = idf.shape[0]            
            batch_size = df_xy_args['batch_size']
            batches = idfs//batch_size
            batch_rows = batch_size*batches
            
            
            for __ in range(0,batches):
                df_xy_args['offset']=__
                imgs, fxs,fys = df_x_y (**df_xy_args)
                iy=np.concatenate([fxs,fys],1)
                ix=imgs
                fdt={xi:ix,y_:iy}
                fd.update(fdt)
                _t,l= s.run([train_step,loss],feed_dict=fd)
                if v==True:
                    print("Iter:",_,"batch",__,"batches",batches,"Loss:",l)
                
            if batch_rows<idfs:
                df_xy_args['offset']=batches
                df_xy_args['batch_size'] = idfs - batch_rows               
                imgs, fxs,fys = df_x_y (**df_xy_args)
                iy=np.concatenate([fxs,fys],1)
                ix=imgs
                
                fdt={xi:ix,y_:iy}
                fd.update(fdt)
                _t,l= s.run([train_step,loss],feed_dict=fd)
                if v==True:
                    print("Iter:",_,"batch",__+1,"batches",batches,"Loss:",l)
            
            
            
        if save_model==True:
            if save_name==None:
                print("No model specified, model not being saved")
                return
            else:
                save_path = saver.save(s, save_name)
                print("Model saved in file: %s" % save_name)

def create_model(ixs,iys,model=None,opt_mode='classification'):    
    if model==None:
        print("No model specified")
        return 0
    tf.reset_default_graph()
    class_output = iys[1]
    d0 = ixs[0]
    x_shape=[None]
    for _ in range(1,len(ixs)):
        x_shape.append(ixs[_])
    xi = tf.placeholder(tf.float32, shape=x_shape,name='x')
    y_ = tf.placeholder(tf.float32, shape=[None,class_output],name='y')
    train_bool=tf.placeholder(bool,name='train_test')
    
    learning_rate = tf.placeholder(tf.float32)
    
    #Define the model here--DOWN
    x = xi
    types_dic = {'conv':0,'bn':0,'relu':0,'max_pool':0,'drop_out':0,'fc':0,'res_131':0}
    for i,_ in enumerate(model):
        _type=_[0]
        _input=x
        params={'_input':_input}
        #print(_)
        if len(_)==2:
            _params=_[1]
            #print(params,_params)
            params.update(_params)
        counter=types_dic[_type]
        types_dic[_type]+=1
        name_scope=_type+str(counter)
        if _type=='conv':    
            x=conv(**params)
        elif _type=='bn':
            params['is_training']=train_bool
            x = batch_norm(**params)
        elif _type=='relu':
            x = relu(**params)
        elif _type=='max_pool':
            x = max_pool(**params)
        elif _type=='drop_out':
            params['is_training']=train_bool
            x = drop_out(**params)
        elif _type=='fc':
            params['pre_conv']=False
            if i>0:
                if model[i-1][0]=='conv':
                    params['pre_conv']=True
            x = fc(**params)
        elif _type=='res_131':
            params['is_training']=train_bool
            x = _res_131(**params)

    prev_conv_fcl = False
    if model[-1][0] in ['conv','res_131']:
        prev_conv_fcl=True
    prediction = fc(x,n=class_output,name_scope="FCL",prev_conv=prev_conv_fcl)
    
    #Define the model here--UP
    
    y_CNN = tf.nn.softmax(prediction,name='Softmax')
    class_pred = tf.argmax(y_CNN,1,name='ClassPred')
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]),name="loss")
    
    if opt_mode=='classification':        
        y_CNN = tf.nn.softmax(prediction,name='Softmax')        
        class_pred = tf.argmax(y_CNN,1,name='ClassPred')       
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]),name="loss")
        acc_,spe_,sen_,tp_,tn_,fp_,fn_ = stats_class(y_CNN,y_)
        stats_dic={'acc':acc_,'spe':spe_,'sen_':sen_,'tp':tp_,'tn':tn_,'fp':fp_,'fn':fn_}
    elif opt_mode=='regression':
        loss = tf.reduce_mean(tf.pow(tf.subtract(y_,prediction),2),name='loss')
        stats_dic={'loss':loss}   
        
    #The following three lines are required to make "is_training" work for normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return [xi,y_,learning_rate,train_bool,loss,train_step,stats_dic]

def predict_model(iX,model_dir=None,opt_mode='classification',is_training=False):
    if model_dir==None:
        print("No model to load")
        return
    else:
        save_dir=model_dir
        with tf.Session('', tf.Graph()) as s:
            with s.graph.as_default():
                saver = tf.train.import_meta_graph(save_dir+".meta")
                saver.restore(s,save_dir)
                fd={'x:0':iX,'train_test:0':is_training}
                if opt_mode=='classification':
                    score,c = s.run(['Softmax:0','ClassPred:0'],feed_dict=fd)
                    
                    return score,c
                elif opt_mode=='regression':
                    fcl = s.run('FCL/FC:0',feed_dict=fd)
                    return fcl

def stats_class(predicted,ground_truth):
    yi = tf.argmax(ground_truth,1)
    yp = tf.argmax(predicted,1)
    tpi = yp*yi
    tp = tf.reduce_sum(tf.cast(tf.greater(tpi,0),tf.int32),name='tp')
    fni = yi-yp
    fn = tf.reduce_sum(tf.cast(tf.greater(fni,0),tf.int32),name='fn')
    sensitivity = tf.divide(tp,(fn+tp),name='sen')    #sensitivity = tp/(fn+tp)    
    tni = yi+yp
    tn = tf.reduce_sum(tf.cast(tf.equal(tni,0),tf.int32),name='tn')    
    fpi = yp - yi
    fp = tf.reduce_sum(tf.cast(tf.greater(fpi,0),tf.int32),name='fp')
    specificity = tf.divide(tn,(tn+fp),name='spe')#specificity = tn/(tn+fp)
    accuracy = tf.divide((tn+tp),(tn+tp+fn+fp),name='acc')#accuracy = (tn+tp)/(tn+tp+fn+fp)
    return [accuracy,specificity,sensitivity,tp,tn,fp,fn]

def acc_sen_spe(tp,tn,fp,fn):
    stats_dic={}
    stats_dic['sen']=(tp/(tp+fn))
    stats_dic['acc']=(tn+tp)/(tp+tn+fp+fn)
    stats_dic['spe']=tn/(tn+fp)
    return stats_dic.copy()

def df_x_y (df,x_label,xp_label,yp_label,batch_size=5,offset=0,resize_wh=(32,32),toGray=False):
    if toGray==True:
        channels=1
    else:
        channels=3
    x_y = df.iloc[offset:offset+batch_size]
    images = x_y[x_label].values
    imgs = []
    fxa = []
    fya = []
    for _ in range(0,batch_size):
        img, fx,fy = imgOpenResize(images[_],resize_wh)
        
        fxa.append(fx)
        fya.append(fy)
        imgs.append(img)
    #zz = list(map(imgOpenResize,fx,[resize_wh for _ in range(0,batch_size) ]))
    
    def div_np(num,ilist,batch_size):
        div_m =  np.asarray([ilist]).reshape((batch_size,1))
        td = div_m
        for _n_ in range(0,num.shape[1]-1):
            div_m = np.concatenate([div_m,td],1)
        return num*div_m
        
    
    xps = div_np(x_y[xp_label],fxa,batch_size)
    yps = div_np(x_y[yp_label],fya,batch_size)
       
    x = np.asarray(imgs)
    if toGray==True:
        x = list(map(npToGray,x))
    x = np.asarray(x)
    x = x.reshape([x.shape[0],x.shape[1],x.shape[2],channels])
    
    return x,xps.values,yps.values

def rgbToG(img):
    """
    Color to gray
    """
    npImg=np.asarray(img)
    r=0.2125
    g=0.7154
    b=0.0721
    gsImg=r*npImg[:,:,0]+g*npImg[:,:,1]+b*npImg[:,:,2]
    return gsImg

def npToGray(iNp):
    """
    Open Image and 
    """
    imnp=iNp
    ims=len(imnp.shape)
    if ims == 3:
        imgG=rgbToG(imnp)
    elif ims ==2:
        imgG=imnp
    return imgG

def imgOpenResize(imgF,w_h = (32,32)):
    """
    Open Image and 
    """
    img=Image.open(imgF)
    img_size = img.size
    xf = w_h[0]/img_size[0]
    yf = w_h[1]/img_size[1]

    img = img.resize(w_h,Image.ANTIALIAS)
    imnp=np.asarray(img)
    img.close() #Close opened image
    return imnp,xf,yf



def plot_image_points(df,_index_,fullname,max_point=11,figsize=(8,8)):
    """
    Given a dataframe df load a picture in a given row:_index_ by the fullname: absolute path
    with max_point the number of points/columns that are present as landmarks.
    """
    file = df[fullname].loc[_index_]
    x_labels=['x'+str(_) for _ in range(0,max_point)]
    y_labels=['y'+str(_) for _ in range(0,max_point)]
    xp = df[x_labels].loc[_index_]
    yp = df[y_labels].loc[_index_]
    print(file)
    ii = Image.open(file)
    ni = np.asarray(ii)
    ii.close()
    plt.figure(figsize=figsize)
    plt.imshow(ni)
    plt.scatter(xp,yp,marker='x',c='b')
    plt.show()
    
def plot_prediction_points(imgs_array,predicted_points,_index_=0,figsize=(8,8)):
    pred_xy = predicted_points[_index_]
    points=pred_xy.shape[0]//2
    xps = pred_xy[0:points]
    yps = pred_xy[points:]
    x = imgs_array[_index_]
    plt.figure(figsize=(8,8))
    plt.imshow(x)
    plt.scatter(xps,yps)
    plt.show()

def _res_131(_input,depths=[4,4,8],
              name_scope="res",is_training=True):
    ims = _input.get_shape().as_list()
    input_depth=ims[len(ims)-1]
    filter_kernel={0:1,1:3}
    with tf.name_scope(name_scope):
        shortcut = _input
        _output = _input
        for i,__ in enumerate(depths):
            _={'filter_size':filter_kernel[i%2],'layer_depth':__}
            _['name_scope']=name_scope+'_C'+str(i)
            _output = conv(_output, **_)
            _output = batch_norm(_output,is_training=is_training)
            rn=name_scope+'_A'+str(i)
            if _==len(depths)-1:
                _output = _output + shortcut
                rn=name_scope+"_A_last"
            _output = relu(_output,name_scope=rn)
    return _output