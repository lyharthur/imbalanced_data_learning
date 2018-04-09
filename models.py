import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import csv
import numpy as np
import matplotlib as mpl
from sklearn.metrics import classification_report

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
import os,sys
import random
from PIL import Image
import scipy.misc
from sklearn.svm import SVC
import tensorflow.contrib.losses as tf_losses

from nets import *
from datas import *
from tfrecord import *
from triplet_loss import *
from t_sne import *

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
 
def sample_y(m, n): # 16 , y_dim , fig count
    y = np.zeros([m,n])
    for i in range(m):
        y[i, i%n] = 1
    #y[:,7] = 1
    #y[-1,0] = 1
    #print(y)
    return y

def concat(z,y):
    return tf.concat([z,y],1)
def eval(confusion_matrix):
    TN = confusion_matrix[0,0] 
    FN = confusion_matrix[0,1] 
    FP = confusion_matrix[1,0] 
    TP = confusion_matrix[1,1] 
    precision = TP / (TP+FN)
    recall = TP / (TP+FP)
    acc = (TP+TN) / (TP+FN+TN+FP)
    F1 = 2*precision*recall/(recall+precision)
    return acc, precision, recall, F1

def evaluation(confusion_matrix):
    acc = (confusion_matrix[0,0] + confusion_matrix[1,1] + confusion_matrix[2,2] + confusion_matrix[3,3] + confusion_matrix[4,4])/np.sum(confusion_matrix)

    sensitivity_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[1,0] + confusion_matrix[2,0] + confusion_matrix[3,0] + confusion_matrix[4,0]) 
    sensitivity_1 = confusion_matrix[1,1] / (confusion_matrix[0,1] + confusion_matrix[1,1] + confusion_matrix[2,1] + confusion_matrix[3,1] + confusion_matrix[4,1]) 
    sensitivity_2 = confusion_matrix[2,2] / (confusion_matrix[0,2] + confusion_matrix[1,2] + confusion_matrix[2,2] + confusion_matrix[3,2] + confusion_matrix[4,2]) 
    sensitivity_3 = confusion_matrix[3,3] / (confusion_matrix[0,3] + confusion_matrix[1,3] + confusion_matrix[2,3] + confusion_matrix[3,3] + confusion_matrix[4,3]) 
    sensitivity_4 = confusion_matrix[4,4] / (confusion_matrix[0,4] + confusion_matrix[1,4] + confusion_matrix[2,4] + confusion_matrix[3,4] + confusion_matrix[4,4])

    specificity_0 = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1] + confusion_matrix[0,2] + confusion_matrix[0,3] + confusion_matrix[0,4]) 
    specificity_1 = confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1] + confusion_matrix[1,2] + confusion_matrix[1,3] + confusion_matrix[1,4]) 
    specificity_2 = confusion_matrix[2,2] / (confusion_matrix[2,0] + confusion_matrix[2,1] + confusion_matrix[2,2] + confusion_matrix[2,3] + confusion_matrix[2,4]) 
    specificity_3 = confusion_matrix[3,3] / (confusion_matrix[3,0] + confusion_matrix[3,1] + confusion_matrix[3,2] + confusion_matrix[3,3] + confusion_matrix[3,4]) 
    specificity_4 = confusion_matrix[4,4] / (confusion_matrix[4,0] + confusion_matrix[4,1] + confusion_matrix[4,2] + confusion_matrix[4,3] + confusion_matrix[4,4]) 

    print('ACC:',acc)
    print('sensitivity:')
    print(sensitivity_0,sensitivity_1,sensitivity_2,sensitivity_3,sensitivity_4)
    print('specificity:')
    print(specificity_0,specificity_1,specificity_2,specificity_3,specificity_4)

    return 

class GAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier, data_all, data_min, data_val):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.data_all = data_all
        self.data_min = data_min
        self.data_val = data_val

        self.lr = tf.placeholder(tf.float32)

        self.lam = 1e-3
        self.gamma = 0.5
        self.k_curr = 0.0


        # data
        self.z_dim = self.data_all.z_dim
        self.y_dim = self.data_all.y_dim # condition
        self.size = self.data_all.size
        self.channel = self.data_all.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.k = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, name='IsTraining')

        # nets
        self.G_sample = self.generator(self.z)

        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)
    
        self.C_real = self.classifier(self.X, is_training=self.is_training)
        self.C_fake = self.classifier(self.G_sample, is_training=self.is_training, reuse = True)

        self.lam = 10
        eps = tf.random_uniform([], minval=0., maxval=1.)#batch_size = 64
        self.X_inter = eps*self.X + (1. - eps)*self.G_sample
        self.D_tmp = discriminator(self.X_inter, reuse = True)
        grad = tf.gradients(self.D_tmp, self.X_inter)[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = self.lam * tf.reduce_mean(grad_norm - 1.)**2    

        # loss
        C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))
        C_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y))
        
        self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake) + grad_pen
        self.G_loss = - tf.reduce_mean(self.D_fake)*0.6 #+ C_fake_loss*0.4
        self.C_loss = C_real_loss*0.6 + C_fake_loss*0.4 
        #self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))  
        #self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]

        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.generator.vars)
        self.C_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.C_loss, var_list=self.classifier.vars)


        self.correct_prediction = tf.equal(tf.argmax(self.C_real, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        argmax_prediction = tf.argmax(self.C_real, 1)
        argmax_y = tf.argmax(self.y, 1)

        #self.TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
        #self.TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
        #self.FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
        #self.FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)

        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(argmax_y, argmax_prediction, num_classes=self.y_dim)
        self.auc = tf.metrics.auc(argmax_y, argmax_prediction)
        #self.C_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.C_fake_loss, var_list=self.generator.vars)        

        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train_classifier(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 32, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 2e-4

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                X_b, y_b = self.data(batch_size)
                self.sess.run(
                    self.C_solver1,
                    feed_dict={self.X: X_b, self.y: y_b, self.lr: learning_rate}
                    )
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                C_real_loss_curr = self.sess.run(
                        [self.C_real_loss],
                        feed_dict={self.X: X_b, self.y: y_b})
                print(epoch,C_real_loss_curr)
                #print('Iter: {}; C_real_loss: {:.4}'.format(epoch,  C_real_loss_curr))

            if epoch % 1000 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)
     

    def train(self, sample_folder, ckpt_dir, training_epoches = 100000, batch_size = 128, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 1e-4
        
        image, label = read_and_decode(self.data_all.filename, batch_size,self.size)
        image_min, label_min = read_and_decode(self.data_min.filename, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data_val.filename, batch_size,self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        for epoch in range(training_epoches):
            
            n_d = 80 if epoch < 30 or (epoch+1) % 500 == 0 else 5
            n_g = 1 if epoch < 30 or (epoch+1) % 500 == 0 else 1
            n_c = 1 if epoch < 1000 or (epoch+1) % 500 == 0 else 40

            # update D
            for _ in range(n_d):
                #X_b, y_b = self.data_min(batch_size)
                X_b, y_b = get_batch(self.sess, image_min, label_min, self.y_dim, batch_size)

                self.sess.run(
                    [self.D_solver], #clip
                    feed_dict={self.X: X_b, self.z: sample_z(X_b.shape[0], self.z_dim), self.lr: learning_rate})
            # update G
            for _ in range(n_g):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.z: sample_z(X_b.shape[0], self.z_dim), self.y: y_b, self.lr: learning_rate, self.is_training: True})
                
            # update C
            for _ in range(n_c):
                #X_b, y_b = self.data_all(batch_size)
                X_b, y_b = get_batch(self.sess, image, label, self.y_dim, batch_size)
                self.sess.run(
                    self.C_solver,
                    feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(X_b.shape[0], self.z_dim), self.lr: learning_rate, self.is_training: True})
            
            # save img, model. print loss
            if epoch % 500 == 0 or epoch <= 100:

                #train
                G_loss_curr, D_loss_curr, C_loss_curr, C_acc_curr ,confusion_matrix= self.sess.run(
                        [self.G_loss, self.D_loss, self.C_loss, self.accuracy, self.confusion_matrix],
                        feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(X_b.shape[0], self.z_dim), self.lr: learning_rate, self.is_training: False})

                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_loss: {:.4}; C_acc: {:.4}'.format(epoch, D_loss_curr, G_loss_curr, C_loss_curr, C_acc_curr))
                print(confusion_matrix)

                #test
                X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
                C_acc_val, confusion_matrix_val = self.sess.run(    
                                [self.accuracy, self.confusion_matrix],
                                feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})
                print(C_acc_val)
                print(confusion_matrix_val)
                #acc, sens, spec= eval(confusion_matrix_val)
                #print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

                if epoch % 500 == 0:
                    y_s = sample_y(16, self.y_dim)
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

                    fig = self.data_min.data2fig(samples)
                    plt.savefig('{}/{}.png'.format(sample_folder, str(fig_count).zfill(3)), bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)

            if epoch % 1000 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'GAN_C.ckpt', global_step=epoch)

    def test(self, sample_folder, sample_num):
        y_s = sample_y(sample_num, self.y_dim)
        samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(sample_num, self.z_dim)})
        for i, sample in enumerate(samples):
            new_sample = np.concatenate((sample,sample),axis = 2)
            new_sample = np.concatenate((new_sample,sample),axis = 2)
            #fig = self.data.data2fig(samples)
            plt.imshow(new_sample)
            plt.axis('off')
            plt.savefig('{}/{}_{}.png'.format(sample_folder, i%self.y_dim, str(i).zfill(3)), bbox_inches='tight')
            plt.close()


    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")


class WGAN():
    def __init__(self, generator, discriminator, data):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self.tanh = True

        self.z_dim = self.data.z_dim
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        # nets
        self.G_sample = self.generator(self.z)

        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)

        # loss
        # improved wgan
        self.lam = 10
        eps = tf.random_uniform([], minval=0., maxval=1.)#batch_size = 64 or lower than 64
        self.X_inter = eps*self.X + (1. - eps)*self.G_sample
        self.D_tmp = discriminator(self.X_inter, reuse = True)
        grad = tf.gradients(self.D_tmp, self.X_inter)[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = self.lam * tf.reduce_mean(grad_norm - 1.)**2
        

        
        self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake) + grad_pen
        self.G_loss = - tf.reduce_mean(self.D_fake)

        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.G_loss, var_list=self.generator.vars)
        
        # clip
        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]
        
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 128, restore = True):
        i = 0
        if not restore:
            self.sess.run(tf.global_variables_initializer())
            #sess_glob.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        image, label = read_and_decode(self.data.filename_train, batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)

        for epoch in range(training_epoches):
            # update D
            n_d = 100 if (restore==False and epoch < 25) or (epoch+1) % 500 == 0 else 5
            #print(n_d)

            for _ in range(n_d):
                #X_b, _ = self.data(batch_size)
                #self.sess.run(self.clip_D)
                #TFrecord
                X_b, _ = get_batch(self.sess, image, label, 1, batch_size)

                self.sess.run(
                        self.D_solver,
                        feed_dict={self.X: X_b, self.z: sample_z(X_b.shape[0], self.z_dim)}
                        )
            # update G
            self.sess.run(
                self.G_solver,
                feed_dict={self.z: sample_z(batch_size, self.z_dim)}
            )

            # print loss. save images.
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr = self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_b, self.z: sample_z(X_b.shape[0], self.z_dim)})
                G_loss_curr = self.sess.run(
                        self.G_loss,
                        feed_dict={self.z: sample_z(X_b.shape[0], self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

                if epoch % 500 == 0:
                    samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(16, self.z_dim)})

                    fig = self.data.data2fig(samples)
                    plt.savefig('{}/{}.jpeg'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)
            if epoch % 1000 == 0 and epoch != 0:
                self.saver.save(self.sess, ckpt_dir+'W_GAN.ckpt', global_step=epoch)
    def test(self, sample_folder, sample_num):
        samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(sample_num, self.z_dim)})
        for i, sample in enumerate(samples):
            #new_sample = np.concatenate((sample,sample),axis = 2) # for gray img
            #new_sample = np.concatenate((new_sample,sample),axis = 2)
            #fig = self.data.data2fig(samples)
            plt.imshow(sample)
            plt.axis('off')
            plt.savefig('{}/{}.jpeg'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
            plt.close()

    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")

class Classifer():
    def  __init__(self, classifier, data):
        self.classifier = classifier
        self.data = data
        

        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.is_training = tf.placeholder(tf.bool, name='IsTraining')

        self.logits = self.classifier(self.X, self.is_training)

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        argmax_prediction = tf.argmax(self.logits, 1)
        argmax_y = tf.argmax(self.y, 1)

        # self.TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
        # self.TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
        # self.FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
        # self.FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
        # self.precision = self.TP / (self.TP + self.FP)
        # self.recall = self.TP / (self.TP + self.FN)

        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(argmax_y, argmax_prediction, num_classes=self.y_dim)
        #self.auc = tf.metrics.auc(argmax_y, argmax_prediction)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y))
        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cross_entropy,  var_list=self.classifier.vars)

        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, ckpt_dir, training_epoches = 8200, batch_size = 256, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 1e-4
        print(self.data.filename_train,self.data.filename_val)
        image, label = read_and_decode(self.data.filename_train, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data.filename_val, batch_size,self.size)

        #image, label = read_and_decode('train_slm_512_01234.tfrecords',batch_size,self.size)
        #image_val, label_val = read_and_decode('test_slm_512_01234.tfrecords',batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                #X_b, y_b = self.data(batch_size)
                X_b, y_b = get_batch(self.sess, image, label, self.y_dim, batch_size)

                self.sess.run(
                    [self.train_step],
                    feed_dict={self.X: X_b, self.y: y_b, self.is_training: True})
            
            # save img, model. print loss
            if epoch % 500 == 0 or epoch <= 10:
                #C_loss_curr, C_acc_curr, TP, TN, FP, FN, precision, recall, confusion_matrix = self.sess.run(
                C_loss_curr, C_acc_curr, confusion_matrix = self.sess.run(	
                        #[self.cross_entropy, self.accuracy, self.TP, self.TN, self.FP, self.FN, self.precision, self.recall,self.confusion_matrix],
                        [self.cross_entropy, self.accuracy, self.confusion_matrix],
                        feed_dict={self.X: X_b, self.y: y_b, self.is_training: False})
                #print(epoch, C_loss_curr, C_acc_curr)
                #print('Iter: {}; C_loss: {:.4}; C_acc: {:.4}  TP: {}; TN: {:}; FP: {:}; FN: {:}; precision: {:}; recall: {:}'.format(epoch, C_loss_curr, C_acc_curr,TP,TN,FP,FN,precision,recall))
                print('Iter: {}; C_loss: {:.4}; C_acc: {:.4};'.format(epoch, C_loss_curr, C_acc_curr))

                #print(''.format(TP,TN,FP,FN,precision,recall))
                # print(confusion_matrix)
                #evaluation(confusion_matrix)
                #acc, sens, spec = eval(confusion_matrix)
                #print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

                # validation
                X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
                y_pred, confusion_matrix_final = self.sess.run(
                            [self.logits, self.confusion_matrix],
                            feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})
                y_val_b_total = np.argmax(y_val_b, axis=1)
                y_pred_total = np.argmax(y_pred, axis=1)

                if epoch % 1000 == 0 :
                    batch_number = self.data.len_val//batch_size
                    #print(batch_number)
                    for _ in range(batch_number):
                        X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
                        y_pred, confusion_matrix_val = self.sess.run(
                                        [self.logits, self.confusion_matrix],
                                        feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})
                        y_val_b = np.argmax(y_val_b, axis=1)
                        y_pred = np.argmax(y_pred, axis=1)

                        y_val_b_total = np.append(y_val_b_total, y_val_b)
                        y_pred_total = np.append(y_pred_total, y_pred)

                        # print(y_val_b_total.shape, y_pred_total.shape)
                        confusion_matrix_final += confusion_matrix_val
                    print(confusion_matrix_final)
                    print(classification_report(y_val_b_total, y_pred_total))
                    #print(Acc_test)
                    # acc, pre, rec, F1 = eval(confusion_matrix_final)
                    # print('Accurancy: {:.4}; precision: {:.4}; recall: {:.4}; F1-score: {:.4}'.format(acc, pre, rec, F1))
                #evaluation(confusion_matrix_final)
                #print('AUC {:.4}'.format(AUC_val))
                #acc, sens, spec = eval(confusion_matrix_final)
                #print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

            if epoch % 1000 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)

    def test(self, data_val, batch_size = 20):
        image_val, label_val = read_and_decode(self.data_val.filename, batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)

        batch_number = self.data_val.len//batch_size
        X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
        confusion_matrix_final = self.sess.run(
                        self.confusion_matrix,
                        feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})
        for _ in range(batch_number-1):
            X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
            confusion_matrix_val = self.sess.run(
                        self.confusion_matrix,
                        feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})

            confusion_matrix_final+=confusion_matrix_val
        print(confusion_matrix_final)
        evaluation(confusion_matrix)

        #acc, sens, spec = eval(confusion_matrix_final)
        #print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")            

class Autoencoder():
    def  __init__(self, autoencoder, data):
        self.autoencoder = autoencoder
        self.data = data

        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
     
        self.mse, self.feature_space = self.autoencoder(self.X)

        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.mse,  var_list=self.autoencoder.vars)

        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, ckpt_dir, training_epoches = 1000000, batch_size = 256, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 1e-4
        print(self.data.filename)
        image, label = read_and_decode(self.data.filename, batch_size,self.size)

        #image, label = read_and_decode('train_slm_512_01234.tfrecords',batch_size,self.size)
        #image_val, label_val = read_and_decode('test_slm_512_01234.tfrecords',batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)
        # for _ in range(1000):
        #     X_b, y_b = get_batch(self.sess, image, label, self.y_dim, batch_size)
        #     X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
        #threads= tf.train.start_queue_runners(sess=self.sess)

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                #X_b, y_b = self.data(batch_size)
                X_b, y_b = get_batch(self.sess, image, label, self.y_dim, batch_size)

                self.sess.run(
                    [self.train_step],
                    feed_dict={self.X: X_b})
            
            # save img, model. print loss
            if epoch % 200 == 0 or epoch <= 100:
                MSE, feature_space = self.sess.run(  
                        [self.mse, self.feature_space],
                        feed_dict={self.X: X_b})
                print('Iter: {}; C_loss: {:.4};'.format(epoch, MSE))
                #print(feature_space)



            if epoch % 100 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)

    def test(self, data, sample_num, batch_size = 1):
        image, label = read_and_decode(self.data.filename, batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)

        if sample_num==0:
            output_num = self.data.len
        else:
            output_num = sample_num
        feature_space_out = [np.zeros(128+1)]## feature dimension
        
        for i in range(output_num):
        
            X_b, y_b = get_batch(self.sess, image, label, self.y_dim, batch_size)

            feature_space_run = self.sess.run(self.feature_space, feed_dict={self.X: X_b})

            #print(feature_space[0], [np.argmax(y_b[0])])
            if np.argmax(y_b[0])>=4:
                feature_space = np.append(feature_space_run, [np.argmax(y_b[0])])
                feature_space_out = np.append(feature_space_out, [feature_space], axis=0)
            feature_space = np.append(feature_space_run, [np.argmax(y_b[0])])
            #print(feature_space)
            feature_space_out = np.append(feature_space_out, [feature_space], axis=0)
            #y_out = np.append(y_out, [np.argmax(y_b[0])], axis=0)

        #print(feature_space_out)
        #print(y_out)
        #np.append(feature_space_out, y_out, axis=1)
        np.savetxt('feature_train.csv', feature_space_out, delimiter=',') 
    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")

class one_shot_learning():
    def  __init__(self, one_shot, data):
        self.one_shot = one_shot
        self.data = data
        #self.data_2 = data_2

        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.feature_size = 256
        self.channel = self.data.channel

        self.X_1 = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.X_2 = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.y_1 = tf.placeholder(tf.float32, shape=[None, 2])#one-hot
        self.y_2 = tf.placeholder(tf.float32, shape=[None, 2])
        self.is_training = tf.placeholder(tf.bool, name='IsTraining')

        #self.y = tf.equal(tf.argmax(self.y_1), tf.argmax(self.y_2))
        self.y = tf.cast(tf.equal(self.y_1, self.y_2), tf.float32)

        self.left, self.left_fc, self.left_mse, self.x_out_l = self.one_shot(self.X_1, self.feature_size, self.is_training)        
        self.right, self.right_fc, self.right_mse, self.x_out_r = self.one_shot(self.X_2, self.feature_size, self.is_training, reuse=True)
        # self.left, self.left_fc = self.one_shot(self.X_1, self.feature_size, self.is_training)        
        # self.right, self.right_fc = self.one_shot(self.X_2, self.feature_size, self.is_training, reuse=True)

        self.margin = 10
            
        self.d = tf.reduce_sum(tf.square(tf.subtract(self.left, self.right)), 1,  keep_dims=True)
        self.d_sqrt = tf.sqrt(self.d)

        loss = (1. - self.y) * tf.square(tf.maximum(0., self.margin - self.d_sqrt)) + (self.y) * self.d # dis+ sim

        self.loss = 0.5 * tf.reduce_mean(loss)
        
        self.pred_R = tf.argmax(self.right_fc, 1)
        self.correct_prediction_l = tf.cast(tf.equal(tf.argmax(self.left_fc, 1), tf.argmax(self.y_1, 1)), tf.float32)
        self.correct_prediction_r = tf.cast(tf.equal(tf.argmax(self.right_fc, 1), tf.argmax(self.y_2, 1)), tf.float32)
        self.accuracy_L = tf.reduce_mean(tf.cast(self.correct_prediction_l, tf.float32))
        self.accuracy_R = tf.reduce_mean(tf.cast(self.correct_prediction_r, tf.float32))

        self.cross_entropy_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.left_fc, labels = self.y_1))
        self.cross_entropy_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.right_fc, labels = self.y_2))
        self.total_loss = self.loss + (self.right_mse + self.left_mse) + self.cross_entropy_left + self.cross_entropy_right

        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.total_loss,  var_list=self.one_shot.vars)
        #self.train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(self.loss,  var_list=self.one_shot.vars)
        #self.train_step = tf.train.RMSPropOptimizer(learning_rate=1e-4,decay=0.9,momentum=0.1).minimize(self.loss,  var_list=self.one_shot.vars)
        #self.train_step = tf.train.MomentumOptimizer(1e-4, 0.7, use_nesterov=True).minimize(self.loss,  var_list=self.one_shot.vars)

        self.saver = tf.train.Saver(max_to_keep=3)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, ckpt_dir, training_epoches = 4200, batch_size = 256, restore = True):
         
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 1e-4
        print(self.data.filename_train)
        image1, label1 = read_and_decode(self.data.filename_train, batch_size,self.size)
        image2, label2 = read_and_decode(self.data.filename_train, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data.filename_val, batch_size,self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                #X_b, y_b = self.data(batch_size)
                X_1, y_1, y_1_one = get_batch(self.sess, image1, label1, self.y_dim, batch_size, one_hot=False)
                X_2, y_2, y_2_one = get_batch(self.sess, image2, label2, self.y_dim, batch_size, one_hot=False)

                _ = self.sess.run(
                        [self.train_step],
                        feed_dict={self.X_1: X_1, self.y_1: y_1_one, self.X_2: X_2, self.y_2: y_2_one, self.is_training: True})
                #print(accuracy_R, accuracy_L)
                #print(test_out4)

            # save img, model. print loss
            if epoch % 200 == 0 or epoch <= 10:
                C_loss_curr, distance, accuracy_L, accuracy_R, mse_L, mse_R, img_out_ = self.sess.run(  
                # C_loss_curr, distance, accuracy_L, accuracy_R = self.sess.run(  
                        [self.total_loss, self.d_sqrt, self.accuracy_L, self.accuracy_R, self.left_mse, self.right_mse,self.x_out_l],
                        feed_dict={self.X_1: X_1, self.y_1: y_1_one, self.X_2: X_2, self.y_2: y_2_one, self.is_training: False})

                print(img_out_.shape)
                for img_num in range(0,10):
                    #print(img_out[img_num,:,:,:].shape)
                    img_out = np.asarray(img_out_)
                    img_out = img_out[img_num,:,:,:]*255.
                    img_out = np.reshape(img_out,(self.size, self.size,3))            
                    scipy.misc.imsave('./out/outfile'+str(img_num)+'.jpg', img_out)

                print('autoencoder', mse_L, mse_R)
                print('Iter: {}; C_loss: {:.4};'.format(epoch, C_loss_curr))
                print(distance[:10])
                y_pred = np.zeros(batch_size)
                y = np.zeros(batch_size)
                for index in range(batch_size):
                    if y_1[index] == y_2[index]:
                        y[index] = 1
                    else:
                        y[index] = 0

                    if distance[index] > self.margin : #  dissimilar
                        y_pred[index] = 0 #1 - y_1[index]
                    else: #similar
                        y_pred[index] = 1 #y_1[index]

                #print(y[:10],y_pred[:10])
                print(confusion_matrix(y, y_pred))

                ############### validation ###############
                X_val, y_val, y_val_one = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size, one_hot=False)


                #d_sqrt, accuracy_L, accuracy_R = self.sess.run([self.d_sqrt, self.accuracy_L, self.accuracy_R],
                d_sqrt, pred = self.sess.run([self.d_sqrt, self.pred_R],
                            feed_dict={self.X_1: X_1, self.y_1: y_1_one, self.X_2: X_val, self.y_2: y_val_one, self.is_training: False})
                print(d_sqrt[:10])
                #print(y_val[:5],pred[:5])
                print('fc')
                print(confusion_matrix(y_val,pred))
                #print(accuracy_L, accuracy_R)
                y_pred = np.zeros(batch_size)
                for index in range(batch_size):
                    if d_sqrt[index] > self.margin:
                        y_pred[index] = 1 - y_1[index]
                    else:
                        y_pred[index] = y_1[index]

                #print(y_val[:5],y_pred[:5])
                confusion_matrix_final = confusion_matrix(y_val, y_pred)
                print('one-shot')
                print(confusion_matrix_final)
                if epoch % 1000 == 0 :
                    batch_number = self.data.len_val//batch_size
                    #print(batch_number)
                    for _ in range(batch_number):
                        X_1, y_1, y_1_one = get_batch(self.sess, image1, label1, self.y_dim, batch_size, one_hot=False)
                        X_val, y_val, y_val_one = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size, one_hot=False)

                        d_sqrt = self.sess.run(self.d_sqrt,
                            feed_dict={self.X_1: X_1, self.y_1: y_1_one, self.X_2: X_val, self.y_2: y_val_one, self.is_training: False})

                        y_pred = np.zeros(batch_size)
                        for index in range(batch_size):
                            if d_sqrt[index] > self.margin:
                                y_pred[index] = 1 - y_1[index]
                            else:
                                y_pred[index] = y_1[index]
                        confusion_matrix_val = confusion_matrix(y_val, y_pred)

                        confusion_matrix_final += confusion_matrix_val
                    print(confusion_matrix_final)
                    acc, pre, rec, F1 = eval(confusion_matrix_final)
                    print('Accurancy: {:.4}; precision: {:.4}; recall: {:.4}; F1-score: {:.4}'.format(acc, pre, rec, F1))

            if epoch % 100 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)

    def get_feature(self, batch_size = 50):
        image_train, label_train = read_and_decode(self.data.filename_train, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data.filename_val, batch_size,self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        #batch_number = self.data_val.len//batch_size
        feature_space_num = self.feature_size
        feature_space_out = [np.zeros(feature_space_num+1)]
        feature_space_out = np.array(feature_space_out)
        for i in range((self.data.len_train_0+self.data.len_train_1)//batch_size):
            X, y, y_one= get_batch(self.sess, image_train, label_train, self.y_dim, batch_size, one_hot=False)
            feature_space = self.sess.run(self.left, feed_dict={self.X_1: X, self.is_training: False})
            #print(y_val_b[:5], feature[:5])
            
            feature_space = np.append(feature_space, y, axis=1)
            feature_space_out = np.append(feature_space_out, feature_space, axis=0)
            #print(feature_space_out[:3])

        #print(feature_space_out)
        #print(y_out)
        #np.append(feature_space_out, y_out, axis=1)
        # sperate 0,1
        feature_space_Maj = []
        feature_space_min = []
        for i in range((self.data.len_train_0+self.data.len_train_1)-batch_size):
            # print(feature_space_out[i+1, -1])
            if feature_space_out[i+1, -1]==0:
                feature_space_Maj.append(feature_space_out[i+1,:])
            if feature_space_out[i+1, -1]==1:
                feature_space_min.append(feature_space_out[i+1,:])
        feature_space_Maj = np.array(feature_space_Maj)
        feature_space_min = np.array(feature_space_min)
        #print(feature_space_min.shape[0])
        feature_mean_Maj = np.mean(feature_space_Maj, axis=0)
        feature_mean_min = np.mean(feature_space_min, axis=0)
        #print(feature_mean_Maj[:10]) 
        #print(feature_mean_min[:10])
        #print(feature_space_min[:3])
        #print(np.subtract(feature_space_min, feature_mean_Maj))
        feature_mean_Maj = np.array(feature_mean_Maj)
        feature_mean_min = np.array(feature_mean_min)
        feature_space_min_new = []
        for _ in range((self.data.len_train_0-self.data.len_train_1)):
            tmp = []
            for i in range(feature_space_num+1):
                r = random.randint(0,feature_space_min.shape[0]-1)
                tmp.append(feature_space_min[r, i])
            tmp = np.array(tmp)
            # print(tmp.shape)
            #distance
            distance_0 = np.sqrt(np.sum(np.square(np.subtract(feature_mean_Maj, tmp))))
            distance_1 = np.sqrt(np.sum(np.square(np.subtract(feature_mean_min, tmp))))
            #print(distance_0,distance_1,distance_0 - distance_1)
            #if distance_0 - distance_1 > 5:
            feature_space_min_new.append(tmp)
        feature_space_min_new = np.array(feature_space_min_new)
        print(feature_space_min_new.shape)
        print(feature_space_Maj.shape)
        print(feature_space_min.shape)
        
        feature_space_all = np.append(feature_space_min_new, feature_space_Maj, axis=0)
        feature_space_all = np.append(feature_space_all, feature_space_min, axis=0)
        np.savetxt('feature_one_shot_train.csv', feature_space_all, delimiter=',') 
        X = feature_space_all[:,:-1]
        y = feature_space_all[:,-1]
        
        #clf = SVC()
        #clf.fit(X,y)
        print('get test')
        # get test
        feature_space_test = [np.zeros(feature_space_num+1)]
        feature_space_test = np.array(feature_space_test)
        print(self.data.len_val)
        for _ in range(self.data.len_val//batch_size):
            X_val, y_val, y_val_one = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size, one_hot=False)
            feature_space = self.sess.run(self.left, feed_dict={self.X_1: X_val, self.is_training: False})
            feature_space = np.append(feature_space, y_val, axis=1)
            feature_space_test = np.append(feature_space_test, feature_space, axis=0)
            #print(_)
        X_test = feature_space_test[1:,:-1]
        y_test = feature_space_test[1:,-1]
        #print(X_test[:5])
        #print(y_test[:5])
        np.savetxt('feature_one_shot_test.csv', feature_space_test, delimiter=',') 

    def test(self, data):
        return 0


    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")            

class triplet_learning():
    def  __init__(self, triplet, data):
        self.triplet = triplet
        self.data = data
        self.min = [0,1,2,3,4]
        #self.data_2 = data_2

        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.feature_size = 64
        self.channel = self.data.channel

        self.X_A = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        # self.X_P = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        # self.X_N = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.y_A = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_A_one = tf.placeholder(tf.float32, shape=[None, self.data.class_num])
        # self.y_P = tf.placeholder(tf.float32, shape=[None, 1])
        # self.y_N = tf.placeholder(tf.float32, shape=[None, 1])
        self.is_training = tf.placeholder(tf.bool, name='IsTraining')

        #self.y = tf.equal(tf.argmax(self.y_1), tf.argmax(self.y_2))

        self.margin = 8

        #self.A_feature, self.A_fc, self.A_mse, self.x_out_a 
        self.A_feature, self.A_mse, self.x_out_a, self.A_fc = self.triplet(self.X_A, self.feature_size, self.is_training)

        # distance_P = tf.reduce_sum(tf.square(A_feature - P_feature), 1)
        # distance_N = tf.reduce_sum(tf.square(A_feature - N_feature), 1)

        # self.loss = tf.reduce_mean(tf.maximum(0., self.margin + distance_P - distance_N))
        
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.A_fc, 1), tf.argmax(self.y_A_one, 1)), tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.A_fc, labels = self.y_A_one))
        
        self.triplet_loss,self.hardest_positive_dist, self.hardest_negative_dist = batch_hard_triplet_loss(self.y_A, self.A_feature, self.margin)
        # self.triplet_loss, _, self.anchor_positive_dist, self.anchor_negative_dist = batch_all_triplet_loss(self.y_A, self.A_feature, self.margin)

        # self.total_loss = self.triplet_loss + self.cross_entropy

        self.total_loss = self.triplet_loss*100. + self.A_mse + self.cross_entropy

        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.total_loss,  var_list=self.triplet.vars)
        #self.train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(self.total_loss,  var_list=self.triplet.vars)
        # self.train_step = tf.train.RMSPropOptimizer(learning_rate=1e-4,decay=0.9,momentum=0.1).minimize(self.total_loss,  var_list=self.triplet.vars)
        #self.train_step = tf.train.MomentumOptimizer(1e-4, 0.7, use_nesterov=True).minimize(self.total_loss,  var_list=self.one_shot.vars)

        self.saver = tf.train.Saver(max_to_keep=3)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, ckpt_dir, training_epoches = 5200, batch_size = 256, restore = True):
         
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 1e-4
        print(self.data.filename_train)
        image1, label1 = read_and_decode(self.data.filename_train, batch_size,self.size)
        # image2, label2 = read_and_decode(self.data.filename_train, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data.filename_val, batch_size,self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                #X_b, y_b = self.data(batch_size)
                X_1, y_1, y_1_one = get_batch(self.sess, image1, label1, self.y_dim, batch_size, one_hot=False)
                # X_2, y_2, y_2_one = get_batch(self.sess, image2, label2, self.y_dim, batch_size, one_hot=False)

                _ = self.sess.run(
                        [self.train_step],
                        feed_dict={self.X_A: X_1, self.y_A: y_1, self.y_A_one: y_1_one, self.is_training: True})
                #print(accuracy_R, accuracy_L)
                #print(test_out4)

            # save img, model. print loss
            if epoch % 200 == 0 or epoch <= 100:
                triplet_L, MSE_L, img_out_, feature, total_loss= self.sess.run(  
                # C_loss_curr, distance, accuracy_L, accuracy_R = self.sess.run(  
                        [self.triplet_loss, self.A_mse, self.x_out_a, self.A_feature, self.total_loss],
                        feed_dict={self.X_A: X_1, self.y_A: y_1, self.y_A_one: y_1_one, self.is_training: False})


                print('triplet:', triplet_L, 'autoencoder', MSE_L)
                print('Iter: {}; C_loss: {:.4};'.format(epoch, total_loss))

                if epoch % 400 == 0: 
                #plot
                    print("Computing t-SNE embedding")           
                    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
                    X_tsne = tsne.fit_transform(feature)

                    plot_embedding(X_tsne, y_1, "t-SNE embedding")       

                    # img out                    
                    # for img_num in range(0,10):
                    #     #print(img_out[img_num,:,:,:].shape)
                    #     img_out = np.asarray(img_out_)
                    #     img_out = img_out[img_num,:,:,:]*255.
                    #     if img_out.shape[2] == 1:
                    #         img_out = np.reshape(img_out,(self.size, self.size))       
                    #     scipy.misc.imsave('./out/outfile'+str(img_num)+'.png', img_out)
                    # plt.show()
                    

    def get_feature(self, batch_size = 50):
        image_train, label_train = read_and_decode(self.data.filename_train, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data.filename_val, batch_size,self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        #batch_number = self.data_val.len//batch_size
        feature_space_num = self.feature_size
        feature_space_out = [np.zeros(feature_space_num+1)]
        feature_space_out = np.array(feature_space_out)
        for i in range((self.data.len_train_0+self.data.len_train_1)//batch_size):
            X, y, y_one= get_batch(self.sess, image_train, label_train, self.y_dim, batch_size, one_hot=False)
            feature_space = self.sess.run(self.A_feature, feed_dict={self.X_A: X, self.is_training: False})
            #print(y_val_b[:5], feature[:5])
            
            feature_space = np.append(feature_space, y, axis=1)
            feature_space_out = np.append(feature_space_out, feature_space, axis=0)
            #print(feature_space_out[:3])

        #print(feature_space_out)
        #print(y_out)
        #np.append(feature_space_out, y_out, axis=1)
        # sperate 0,1
        feature_space_min = []
        feature_space_Maj = []
        for i in range((self.data.len_train_0+self.data.len_train_1)-batch_size):
            # print(feature_space_out[i+1, -1])

            if feature_space_out[i+1, -1] in self.min:
                feature_space_min.append(feature_space_out[i+1,:])
            else:
                feature_space_Maj.append(feature_space_out[i+1,:])
            # if feature_space_out[i+1, -1]==1:

        feature_space_Maj = np.array(feature_space_Maj)
        feature_space_min = np.array(feature_space_min)
        #print(feature_space_min.shape[0])
        feature_mean_Maj = np.mean(feature_space_Maj, axis=0)
        feature_mean_min = np.mean(feature_space_min, axis=0)
        #print(feature_mean_Maj[:10])
        #print(feature_mean_min[:10])
        #print(feature_space_min[:3])
        #print(np.subtract(feature_space_min, feature_mean_Maj))
        feature_mean_Maj = np.array(feature_mean_Maj)
        feature_mean_min = np.array(feature_mean_min)
        feature_space_min_new = []
        # for _ in range((self.data.len_train_1)*5):
        for _ in range(1000):
            tmp = []
            for i in range(feature_space_num+1):
                r = random.randint(0,feature_space_min.shape[0]-1)
                tmp.append(feature_space_min[r, i])
            tmp = np.array(tmp)
            # print(tmp.shape)
            #distance
            # distance_0 = np.sqrt(np.sum(np.square(np.subtract(feature_mean_Maj, tmp))))
            # distance_1 = np.sqrt(np.sum(np.square(np.subtract(feature_mean_min, tmp))))
            #print(distance_0,distance_1,distance_0 - distance_1)
            #if distance_0 - distance_1 > 5:
            feature_space_min_new.append(tmp)
        feature_space_min_new = np.array(feature_space_min_new)
        print(feature_space_min_new.shape)
        print(feature_space_Maj.shape)
        print(feature_space_min.shape)
        
        feature_space_all = np.append(feature_space_min_new, feature_space_Maj, axis=0)
        feature_space_all = np.append(feature_space_all, feature_space_min, axis=0)
        np.savetxt('feature_one_shot_train.csv', feature_space_all, delimiter=',') 
        X = feature_space_all[:,:-1]
        y = feature_space_all[:,-1]
        
        #clf = SVC()
        #clf.fit(X,y)
        print('get test')
        # get test
        feature_space_test = [np.zeros(feature_space_num+1)]
        feature_space_test = np.array(feature_space_test)
        print(self.data.len_val)
        for _ in range(self.data.len_val//batch_size):
            X_val, y_val, y_val_one = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size, one_hot=False)
            feature_space = self.sess.run(self.A_feature, feed_dict={self.X_A: X_val, self.is_training: False})
            feature_space = np.append(feature_space, y_val, axis=1)
            feature_space_test = np.append(feature_space_test, feature_space, axis=0)
            #print(_)
        X_test = feature_space_test[1:,:-1]
        y_test = feature_space_test[1:,-1]
        #print(X_test[:5])
        #print(y_test[:5])
        np.savetxt('feature_one_shot_test.csv', feature_space_test, delimiter=',') 

    def test(self, data):
        return 0


    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")            
