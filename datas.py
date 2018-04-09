import os,sys
import tensorflow as tf
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from tensorflow.examples.tutorials.mnist import input_data

from tfrecord import *

#data_folder = '/home/project/itriDR/pre_data/train_slm_512/'
def get_img(img_path, resize_h):
    img=scipy.misc.imread(img_path).astype(np.float) # mode L for grayscale
    #print(img.shape)
     
    #print(img.shape)
    #crop resize  Original Use
    resize_h = resize_h
    resize_w = resize_h
    #h, w = img.shape[:2]
    #j = int(round((h - crop_h)/2.))
    #i = int(round((w - crop_w)/2.))
    #cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])# cropp
    img = scipy.misc.imresize(img,[resize_h, resize_w])# no cropp
    #print(cropped_image.shape)
    #img = cropped_image#for class grayscale data 
    #img = np.dstack((cropped_image,cropped_image))[:,:,:1]
    #img = cropped_image.reshape((resize_h,resize_w,3))
    #print(img)
    #print(img.shape)
    return np.array(img)/255.0


class mydata():
    def __init__(self, data_folder, size, classes, class_num, is_val=False):

        self.z_dim = 512
        self.y_dim = class_num
        self.size = size
        self.class_num = class_num
        self.channel = 3 ##
        
        #old version?
        data_list = []
        new_id = ''
        for i in range(self.y_dim):
            class_sub = classes.split(',')[i]
            new_id += class_sub
            #print(class_sub)
            datapath = data_folder+class_sub+'/*' #HERE
            print(datapath)
            data_list.extend(glob(datapath))
        print('data_num:',len(data_list))

        self.filename_train = './tfrecords/'+ data_folder.split('/')[2] + '-' + data_folder.split('/')[3]+'-'+new_id+'_train.tfrecords'
        self.filename_val = './tfrecords/'+ data_folder.split('/')[2] + '-' + data_folder.split('/')[3]+'-'+new_id+'_val.tfrecords'

        print(self.filename_train)
        #TFRecord
        self.len_train_0, self.len_train_1, self.len_val = create_TFR(classes,class_num, #
                                    filename_train=self.filename_train,filename_val=self.filename_val, #
                                    folder=data_folder,img_size=self.size,is_val=is_val)


        print(self.len_train_0, self.len_train_1, self.len_val)
        '''
        #get_img
        img_list = [get_img(img_path, self.size) for img_path in data_list]
        self.data = img_list

        #datapath = data_folder+'data/'+class+'/'
        
    
        #self.data = glob(os.path.join(datapath, '*.jpg'))
        #print(self.data)

        # old version?
        label = [] 
        check = []
        label_count = -1
        for path in data_list: 
            class_id = path.split('/')[6]#7,5
            #print(class_id)
            if class_id not in check:
                check.append(class_id)
                label_count+=1
            label.append(label_count)

        #print(label)
        one_hot = np.zeros((len(label),self.y_dim))###self.y_dim  # my im gan = 2
        for i,val in enumerate(label):
            one_hot[i,val]=1
        #print(one_hot)
        self.label = one_hot
        #print(len(label)) 
        self.batch_count = 0

        tmp = list(zip(self.data, self.label))
        random.shuffle(tmp)
        self.data ,self.label = zip(*tmp)'''

    def __call__(self,batch_size):
        #print(img_b)
        #old version
        batch_number = len(self.data)/batch_size
        if self.batch_count < batch_number-1:
            self.batch_count += 1
        else:
            self.batch_count = 0

            tmp = list(zip(self.data, self.label))
            random.shuffle(tmp)
            self.data ,self.label = zip(*tmp)

        img_b = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]
        label_b = self.label[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

        #batch = [get_img(img_path, self.size*3, self.size) for img_path in path_list]
        img_b = np.array(img_b).astype(np.float32)
        
       

        '''
        print self.batch_count
        fig = self.data2fig(batch_imgs[:16,:,:])
        plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        '''
        #print(len(label_list))
        return img_b, label_b

    def data2fig(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
   
        for i, sample in enumerate(samples):
            #print(sample.shape)
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            #print(sample.shape)
            #new_sample = np.concatenate((sample,sample),axis = 2) ## for 1d to 3d
            #new_sample = np.concatenate((new_sample,sample),axis = 2) ## for 1d to 3d
            #print(new_sample.shape)
            #sample = new_sample ## for 1d to 3d
            plt.imshow(sample)
        return fig

class mnist():
    def __init__(self, flag='conv', is_tanh = False):
        datapath = folder+'GAN_yhliu/MNIST_data'
        self.X_dim = 784 # for mlp
        self.z_dim = 100
        self.y_dim = 10
        self.size = 28 # for conv
        self.channel = 1 # for conv
        self.data = input_data.read_data_sets(datapath, one_hot=True)
        self.flag = flag
        self.is_tanh = is_tanh

    def __call__(self,batch_size):
        batch_imgs,y = self.data.train.next_batch(batch_size)
        if self.flag == 'conv':
            batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
        if self.is_tanh:
            batch_imgs = batch_imgs*2 - 1        
        return batch_imgs, y

    def data2fig(self, samples):
        if self.is_tanh:
            samples = (samples + 1)/2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
        return fig    

if __name__ == '__main__':
    data = face3D()
    print(data(17).shape)
