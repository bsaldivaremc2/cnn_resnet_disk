import tensorflow as tf
import numpy as np
import pandas as pd

def get_previous_features(i_layer):
    convx_dims = i_layer.get_shape().as_list()
    return np.prod(convx_dims[1:])

def conv(_input,filter_size=3,layer_depth=8,strides=[1,1,1,1],padding='SAME',
              name_scope="CL",stddev_n = 0.05):
    with tf.name_scope(name_scope):
        ims = _input.get_shape().as_list()
        input_depth=ims[len(ims)-1]
        W = tf.Variable(tf.truncated_normal([filter_size,filter_size,input_depth,layer_depth], stddev=stddev_n),name='W')
        b = tf.Variable(tf.constant(stddev_n, shape=[layer_depth]),name='b')
        c = tf.add(tf.nn.conv2d(_input, W, strides=strides, padding=padding),b,name='conv')
    return c
def batch_norm(_input,is_training=True,decay=0.5,name_scope='BN'):
    with tf.name_scope(name_scope):
        o = tf.contrib.layers.batch_norm(_input, center=True, scale=True, is_training=is_training,decay=decay)
    return o
def drop_out(_input,prop,is_training=True,name_scope='drop_out'):
    with tf.name_scope(name_scope):
        if is_training==True:
            do_prop=prop
        else:
            do_prop=1.0
        _out = tf.nn.dropout(_input, do_prop)
    return _out
def max_pool(_input,kernel=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name_scope='MP'):
    with tf.name_scope(name_scope):
        o = tf.nn.max_pool(_input, ksize=kernel,strides=strides, padding=padding,name='max')
    return o
def relu (_input,name_scope="A"):
    with tf.name_scope(name_scope):
        o = tf.nn.relu(_input,name="activation")
    return o
def fc(_input,n=22,prev_conv=False,
       stddev_n = 0.05,is_training=True,
       name_scope='FC',ecay=0.5):
    with tf.name_scope(name_scope):
        cvpfx = get_previous_features(_input)
        if prev_conv==True:
            im = tf.reshape(_input, [-1, cvpfx])
        else:
            im = _input
        W = tf.Variable(tf.truncated_normal([cvpfx, n], stddev=stddev_n),name='W')
        b = tf.Variable(tf.constant(stddev_n, shape=[n]),name='b') 
        out_ = tf.add(tf.matmul(im, W),b,name="FC")
    return out_
def _res_same(_input,conv_params=[{'filter_size':3},
                                  {'filter_size':3}],
              name_scope="res",is_training=True):
    ims = _input.get_shape().as_list()
    input_depth=ims[len(ims)-1]
    with tf.name_scope(name_scope):
        shortcut = _input
        _output = _input
        for i,_ in enumerate(conv_params):
            _['layer_depth']=input_depth
            _['name_scope']=name_scope+'_C'+str(i)
            _output = conv(_output, **_)
            _output = batch_norm(_output,is_training=is_training)
            rn=name_scope+'_A'+str(i)
            if _==len(conv_params)-1:
                _output = _output + shortcut
                rn=name_scope+"_A_last"
            _output = relu(_output,name_scope=rn)
    return _output

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

