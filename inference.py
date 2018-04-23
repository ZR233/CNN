import os
import tensorflow as tf
import numpy as np


class Config(object):
    # 时间层数
    time_steps = 28*3
    # 隐含层节点数
    num_units = 128
    # 输入维数
    n_input = 28
    # 学习率
    learning_rate = 0.001
    # 类别数
    n_classes = 10

    # 标准图像大小
    pic_size = 32
    # 一批的数量
    batch_size = 128
    # 训练集文件夹
    train_file_path = 'train'


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
    image = tf.image.decode_png(image, 3)
    regular_image = image_dispose(Config, image)
    X, Y = tf.train.batch(
        [regular_image, labels],
        shapes=[[Config.pic_size, Config.pic_size, 3], [Config.n_classes]],
        batch_size=Config.batch_size,
        num_threads=4,
        capacity=Config.batch_size*8)
    return X, Y
