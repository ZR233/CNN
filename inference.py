import os
import tensorflow as tf
class Config(object):
    #时间层数
    time_steps = 28*3
    #隐含层节点数
    num_units=128
    #输入维数
    n_input=28
    #学习率
    learning_rate=0.001
    #类别数
    n_classes=10

    #标准图像大小
    pic_size = 32
    #一批的数量
    batch_size=128

def image_dispose(parameter_list):

    pass




def batch_from_imge(Config):
    imagepaths = []  
    labels = []  
    for c in range(n_classes):
        c_dir = os.path.join(self._file_name,'train', str(c))  
        for parent, dirnames, filenames in os.walk(c_dir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
            filenames = filenames
            parent = parent
            for filename in filenames:                        #输出文件信息
                imagepaths.append(os.path.join(parent, filename))
                label = np.zeros(n_classes)
                label[c] = 1
                labels.append(label.tolist())
    imagepaths = tf.convert_to_tensor(imagepaths, tf.string)  
    labels = tf.convert_to_tensor(labels, tf.float32)
    # 建立 Queue  
    imagepath, self.label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)  

    # 读取图片，并进行解码  
    image = tf.read_file(imagepath)  
    # if re.findall(r'jpg$', imagepath):
    #     image = tf.image.decode_jpeg(image)
    # if re.findall(r'png$', imagepath):
    #     image = tf.image.decode_png(image)
    image = tf.image.decode_png(image,3)
    # 对图片进行裁剪和正则化（将数值[0,255]转化为[-1,1]）  
    image = tf.image.resize_images(image, size=[self._shape[0], self._shape[1]])  
    image = image*1.0/127.5 - 1.0  
    channels = tf.unstack(image,axis= 2)
    channels_reshaped = []
    for i in range(3):
        channels_reshaped.append(tf.reshape(channels[i],[1,-1]))
    self.channels_in_one = tf.concat(channels_reshaped,axis = 1)
    pass