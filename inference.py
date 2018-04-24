# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np


class Config(object):
    # 第一层卷积层过滤器尺寸
    conv1_size = 5
    # 第一层卷积层过滤器深度
    conv1_deep = 32
    # 第二层卷积层过滤器尺寸
    conv2_size = 5
    # 第二层卷积层过滤器深度
    conv2_deep = 32
    # 全连接层节点数
    fc_size = 512
    # 输入维数
    n_input = 28
    # 学习率
    learning_rate = 0.001
    # 类别数
    n_classes = 10

    # 标准图像大小
    pic_size = 32
    # 图像通道数
    pic_channel = 3
    # 一批的数量
    batch_size = 128
    # 训练集文件夹
    train_file_path = 'train'

    # 模型文件路径
    model_path = "model"
    # 模型文件名
    model_name = "model.ckpt"


def image_dispose(Config, image):
    '''图片正则化，返回[size,size,channel]'''
    # 对图片进行裁剪和正则化（将数值[0,255]转化为[-1,1]）
    # 图片转化为(size,size)
    image_shape = image.get_shape()
    if image_shape[0] <= image_shape[1]:
        p = Config.pic_size/(image_shape[0])*(image_shape[1])
        resized = tf.image.resize_images(
            image, (Config.pic_size, int(p)), method=1)
    else:
        p = Config.pic_size/(image_shape[1])*(image_shape[0])
        resized = tf.image.resize_images(
            image, (int(p), Config.pic_size), method=1)
    croped = tf.image.resize_image_with_crop_or_pad(
        resized, Config.pic_size, Config.pic_size)
    return croped


def batch_from_imge(Config):
    imagepaths = []
    labels = []
    for c in range(Config.n_classes):
        c_dir = os.path.join(Config.train_file_path, str(c))
        # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for parent, _, filenames in os.walk(c_dir):
            filenames = filenames
            parent = parent
            for filename in filenames:  # 输出文件信息
                imagepaths.append(os.path.join(parent, filename))
                label = np.zeros(Config.n_classes)
                label[c] = 1
                labels.append(label.tolist())
    imagepaths = tf.convert_to_tensor(imagepaths, tf.string)
    labels = tf.convert_to_tensor(labels, tf.float32)
    # 建立 Queue
    imagepath, label = tf.train.slice_input_producer(
        [imagepaths, labels], shuffle=True)

    # 读取图片，并进行解码
    image = tf.read_file(imagepath)
    image = tf.image.decode_png(image, Config.pic_channel)
    regular_image = image_dispose(Config, image)
    X, Y = tf.train.batch(
        [regular_image, labels],
        shapes=[[Config.pic_size, Config.pic_size, Config.pic_channel], [Config.n_classes]],
        batch_size=Config.batch_size,
        num_threads=4,
        capacity=Config.batch_size*8)
    return X, Y

def inference(input_tensor, train, Config,regularizer):
    with tf.variable_scope('layer1-conv1') :
        conv1_weights = tf.get_variable(
            "weight", [Config.conv1_size, Config.conv1_size, Config.pic_channel, Config.conv1_deep],
            initializer= tf.truncated_normal_initializer(stddev= 0.1))
        conv1_biases = tf.get_variable('biase', [Config.conv1_deep],
            initializer= tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights,strides = [1,1,1,1],
        padding = 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope('layer2-pool1') :
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    # 输入为[size/2,size/2,conv1_deep]
    with tf.variable_scope('layer3-conv2') :
        conv2_weights = tf.get_variable(
            "weight", [Config.conv2_size, Config.conv2_size,Config.conv1_deep, Config.conv2_deep],
            initializer= tf.truncated_normal_initializer(stddev= 0.1))
        conv2_biases = tf.get_variable('biase', [Config.conv2_deep],
            initializer= tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights,strides = [1,1,1,1],
        padding = 'SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    with tf.variable_scope('layer4-pool2') :
        pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    pool_shape = pool2.get_shape().as_list()
    #pool_shap[0]为batch
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0],nodes])

    #全连接层
    with tf.variable_scope('layer5-fc1') :
        fc1_weights = tf.get_variable(
            "weight", [nodes, Config.fc_size],
            initializer= tf.truncated_normal_initializer(stddev= 0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        
        fc1_biases = tf.get_variable('biase', [Config.fc_size],
            initializer= tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer6-fc2') :
        fc2_weights = tf.get_variable(
            "weight", [Config.fc_size, Config.n_classes],
            initializer= tf.truncated_normal_initializer(stddev= 0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        
        fc2_biases = tf.get_variable('biase', [Config.n_classes],
            initializer= tf.constant_initializer(0.1))

        fc2 =tf.matmul(fc1,fc2_weights) + fc2_biases
        
        return fc2