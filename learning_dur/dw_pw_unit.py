import tensorflow as tf 
import tensorflow.contrib.slim as slim 



def depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier, sc, downsample=False):

    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    
    if downsample:
        _strides = 2
    else:
        _strides = 1

    depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=_strides, depth_multiplier=1, kernel_size=[3,3], scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')

    pointwise_conv = slim.convolution2d(bn, num_pwc_filters, kernel_size=[1,1], scope=sc+'/pointwise_conv')

    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')

    return bn

