import numpy as np
import sys
import os
import urllib
from ops import relu, conv2d, deconv2d
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from scipy.misc import imsave,imread

download_url="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"

def download_model(download_link, file_name, expected_bytes=0):
    if os.path.exists(file_name):
        print "VGG19 model already exists"
        return

    file_name,_ = urllib.urlretrieve(download_link, file_name)
    #if os.stat(file_name).st_size == expected_bytes :
    #    print "Download finished"
    #    return 
    #print "Download fail"
    return

def _conv2d_relu(vgg_layers, prev_layer, layer, name):

    W , b = _get_layer(vgg_layers, layer, name)
    with tf.variable_scope(name):
        W = tf.constant(W , name='weights')
        b = tf.constant(b , name='bias')
        conv = tf.nn.conv2d(prev_layer, W, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b)
        
        return relu(conv)

    
def _max_pool(prev_layer, name):
    return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], 
                            strides=[1, 2, 2, 1], padding='SAME', name=name)

def _get_layer(vgg_layers, layer, expected_name):
    
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    name = vgg_layers[0][layer][0][0][0][0]
    #assert name == expected_name
    return W , b.reshape(b.size)

def vgg19(input_image, mat_file='vgg19model.mat'):
    
    """
    Build vgg19 model with pretrained weight
    """
    download_model(download_url, mat_file)
    vgg_layers = loadmat(mat_file)['layers']
    
    vgg_model = {}  
    vgg_model['conv1_1']    = _conv2d_relu(vgg_layers, input_image, 0, 'conv1_1')
    vgg_model['conv1_2']    = _conv2d_relu(vgg_layers, vgg_model['conv1_1'], 2, 'conv1_2')
    vgg_model['maxpool_1']  = _max_pool(vgg_model['conv1_2'], 'conv1_2')
    
    vgg_model['conv2_1']    = _conv2d_relu(vgg_layers, vgg_model['maxpool_1'], 5, 'conv2_1')
    vgg_model['conv2_2']    = _conv2d_relu(vgg_layers, vgg_model['conv2_1'], 7, 'conv2_2')
    vgg_model['maxpool_2']  = _max_pool(vgg_model['conv2_2'], 'maxpool_2')
    
    vgg_model['conv3_1']    = _conv2d_relu(vgg_layers, vgg_model['maxpool_2'], 10, 'conv3_1')
    vgg_model['conv3_2']    = _conv2d_relu(vgg_layers, vgg_model['conv3_1'], 12, 'conv3_2')
    vgg_model['conv3_3']    = _conv2d_relu(vgg_layers, vgg_model['conv3_2'], 14, 'conv3_3')
    vgg_model['conv3_4']    = _conv2d_relu(vgg_layers, vgg_model['conv3_3'], 16, 'conv3_4')
    vgg_model['maxpool_3']  = _max_pool(vgg_model['conv3_4'], 'maxpool_3')

    vgg_model['conv4_1']    = _conv2d_relu(vgg_layers, vgg_model['maxpool_3'], 19, 'conv4_1')
    vgg_model['conv4_2']    = _conv2d_relu(vgg_layers, vgg_model['conv4_1'], 21, 'conv4_1')
    vgg_model['conv4_3']    = _conv2d_relu(vgg_layers, vgg_model['conv4_2'], 23, 'conv4_3')
    vgg_model['conv4_4']    = _conv2d_relu(vgg_layers, vgg_model['conv4_3'], 25, 'conv4_4')
    vgg_model['maxpool_4']  = _max_pool(vgg_model['conv4_4'], 'maxpool_4')

    vgg_model['conv5_1']    = _conv2d_relu(vgg_layers, vgg_model['maxpool_4'], 28, 'conv5_1')
    vgg_model['conv5_2']    = _conv2d_relu(vgg_layers, vgg_model['conv5_1'], 30, 'conv5_2')
    vgg_model['conv5_3']    = _conv2d_relu(vgg_layers, vgg_model['conv5_2'], 32, 'conv5_3')
    vgg_model['conv5_4']    = _conv2d_relu(vgg_layers, vgg_model['conv5_3'], 34, 'conv5_4')
    vgg_model['maxpool_5']  = _max_pool(vgg_model['conv5_3'], 'maxpool_5')

    return vgg_model

IMAGE_NET_MEAN = [123.68, 116.779, 103.939]

def fcn32s(input_image, batch_size):

    mean = tf.constant(IMAGE_NET_MEAN)
    input_image -= mean
    fcn = vgg19(input_image)
    fcn['fc6']      = conv2d(fcn['maxpool_5'], 4096, name='fc6', k_w=7, k_h=7)
    fcn['relu6']    = relu(fcn['fc6'])
    fcn['drop6']    = tf.nn.dropout(fcn['relu6'], 0.5)
    
    fcn['fc7']      = conv2d(fcn['drop6'], 4096, name='fc7', k_w=1, k_h=1)
    fcn['relu7']    = relu(fcn['fc7'])

    #num_class = 6
    fcn['score_fr'] = conv2d(fcn['relu6'], 6, name='score_fr', k_w=1, k_h=1) 
    fcn['upscore']  = deconv2d(fcn['score_fr'], [batch_size, 224, 224, 6], name='upscore')
    fcn['predict']  = tf.argmax(fcn['upscore'], axis=3, name='predict')
    fcn['predict']  = tf.reshape(fcn['predict'], shape=[batch_size, 224, 224, 1])
    return fcn

def fcn8s(input_image, batch_size, ground_truth, learning_rate, IMAGE_SIZE=224):

    input_height = input_width = IMAGE_SIZE
    mean = tf.constant(IMAGE_NET_MEAN)
    input_image -= mean
    fcn = vgg19(input_image)
    fcn['fc6']      = conv2d(fcn['maxpool_5'], 4096, name='fc6', k_w=7, k_h=7)
    fcn['relu6']    = relu(fcn['fc6'])
    fcn['drop6']    = tf.nn.dropout(fcn['relu6'], 0.5)
    
    fcn['fc7']      = conv2d(fcn['drop6'], 4096, name='fc7', k_w=1, k_h=1)
    fcn['relu7']    = relu(fcn['fc7'])
    #num_class = 6

    fcn['score_fr'] = conv2d(fcn['relu6'], 6, name='score_fr', k_w=1, k_h=1) 
    #fcn['upscore']  = deconv2d(fcn['score_fr'], [batch_size, 224, 224, 6], name='upscore')
    #fcn['predict']  = tf.argmax(fcn['upscore'], axis=3, name='predict')
    #fcn['predict']  = tf.reshape(fcn['predict'], shape=[batch_size, 224, 224, 1])

    #print "building"
    fcn['upscore2']    = deconv2d(fcn['score_fr'], [batch_size, 14, 14, 6], name='upscore2', stride_h=2, stride_w=2, k_h=4, k_w=4)
    fcn['score_pool4'] = conv2d(fcn['maxpool_4'], 6, name='score_pool4', k_w=1, k_h=1)
    fcn['fuse_pool4']  = fcn['upscore2'] + fcn['score_pool4']

    #print "building2"
    fcn['upscore_pool4']    = deconv2d(fcn['fuse_pool4'], [batch_size, 28, 28, 6], name='upscore_pool4', stride_h=2, stride_w=2, k_h=4, k_w=4)
    fcn['score_pool3']      = conv2d(fcn['maxpool_3'], 6, name='score_pool3', k_w=1, k_h=1)
    fcn['fuse_pool3']       = fcn['score_pool3'] + fcn['upscore_pool4']

    #print "building22"
    fcn['upscore']          = deconv2d(fcn['fuse_pool3'], [batch_size, 224, 224, 6], name='upscore8', stride_h=8, stride_w=8, k_h=16, k_w=16)
    fcn['predict']          = tf.argmax(fcn['upscore'], axis=3, name='predict8')
    fcn['predict']          = tf.reshape(fcn['predict'], shape=[batch_size, 224, 224, 1])

    #print "buidling22222"
     	
    gts = tf.reshape(ground_truth, shape=[batch_size, input_height, input_width])
    gts = tf.cast(gts, dtype=tf.int32)
    fcn['loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            labels=gts, logits=fcn['upscore']))

    fcn['optimizer'] = tf.train.AdamOptimizer(self.learning_rate, beta1=0.005).minimize(fcn['loss'])
    fcn['var_list']  = tf.trainable_variables()
    fcn['summary_op'] = tf.summary.scalar('loss', self.model['loss'])


    return fcn


#print vgg19('vgg19model.mat')
#download_model(download_url, "vgg19model.mat", 534904783)

COLOR_2_LABEL = np.array([[255, 255, 255] , [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]])

def get_label_id(rgb):
    for class_id in xrange(0, 6):
        if np.array_equal(rgb, COLOR_2_LABEL[class_id]):
            return class_id

def get_rgb(labels):
    return COLOR_2_LABEL[labels]

def labels_2_rgb(label_image):
    "Label image size [width, height, 1]"
    r = np.zeros_like(label_image)
    g = np.zeros_like(label_image)
    b = np.zeros_like(label_image)

    for i in xrange(0, 6):
        r[label_image == i] = COLOR_2_LABEL[i , 0]
        g[label_image == i] = COLOR_2_LABEL[i , 1]
        b[label_image == i] = COLOR_2_LABEL[i , 2]
    
    #rgb = np.zeros((r.shape[0], r.shape[1], 3)).astype(np.uint8)
    return np.concatenate((r, g, b), axis=2)

def rgb_2_labels(rgb_image):

    rgb_image = np.asarray(rgb_image)
    shape = rgb_image.shape
    labels_tensor = np.zeros((shape[0], shape[1], 1))
    
    for r in xrange(shape[0]):
        for c in xrange(shape[1]):
            label_id = get_label_id(rgb_image[r][c])
            labels_tensor[r][c][0] = label_id

    labels_tensor.astype(np.float32)
    return labels_tensor

def get_score(ground_truth, prediction):

    #only Overall Accurracy
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    temp = ground_truth.shape
    sz = 1
    for i in temp:
        sz = sz * i
    #print sz, np.sum((ground_truth == prediction))
    return np.sum((ground_truth == prediction)) * 1.0 / sz

def image_save(imgs , path):
    imsave(path, imgs)

def softmax(arr, axis=-1):
    """
    Return softmax probabilties for given arr and axis
    """
    arr = arr - np.expand_dims(np.max(arr, axis), axis)
    arr = np.exp(arr)
    arr_sum = np.expand_dims(np.sum(arr, axis), axis)
    return arr/arr_sum
    
if __name__ == '__main__' :
    """
    a = imread('test_sample.tif')
    labels = rgb_2_labels(a)
    rev_a = labels_2_rgb(labels)
    imsave('rev_test_sample.tif', rev_a)
    """
    a = np.array([[1 , 2], [2 , 3]])
    b = np.array([[1 , 3], [3 , 4]])
    print get_score(a , b)