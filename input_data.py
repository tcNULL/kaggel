import tensorflow as tf
import os 
import numpy as np
from os import walk
import time
from os.path import join
from scipy.misc import imread,imresize
import pandas as pd

def get_files(file_dir):
    filenames=next(walk(file_dir))[2]  #遍历目录
    num_files=len(filenames)
    print("image number:",num_files)
    image_list = []
    label_list = []
    dic = {}
    f=open('G:/kaggel-cancer-detection/train_labels.csv')
    for i in range(num_files):
        lines=f.readline().split(",")
        dic.setdefault(lines[0],lines[1])  
        if i == num_files:
            break
    for i,filename in enumerate(filenames):
        image_list.append(file_dir + filename)
        filename_len = len(filename)
        filename = filename[0:filename_len - 4]
        label_list.append(int(dic[filename]))
    f.close()
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
	# 打乱顺序
    np.random.shuffle(temp)

	# 取出第一个元素作为 image 第二个元素作为 label
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]  
    return image_list,label_list

def get_batch(image,label,image_W,image_H,batch_size,capacity):
	image = tf.cast(image,tf.string)
	label = tf.cast(label, tf.int32)
	input_queue = tf.train.slice_input_producer([image,label])
	label = input_queue[1]
	# 读取图片
	image_contents = tf.read_file(input_queue[0])
	
	# 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1
	image = tf.image.decode_jpeg(image_contents,channels =3)
	
	# 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
	#image=imresize(image,(96,96))
	image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
	
	#标准化
	image = tf.image.per_image_standardization(image)

	# 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数  tf.train.shuffle_batch 打乱顺序，
	image_batch, label_batch = tf.train.batch([image, label],batch_size = batch_size, num_threads = 64, capacity = capacity)

    # 重新定义下 label_batch 的形状
	label_batch = tf.reshape(label_batch , [batch_size])
	# 转化图片
	image_batch = tf.cast(image_batch,tf.float32)
	return  image_batch, label_batch