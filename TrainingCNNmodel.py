#Deep LearningによるCNNモデルの学習
'''
Created on 2022/03/12

@author: K.ABE
'''

import glob
import numpy as np
import chainer
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import mnist
from chainercv.transforms import resize
from chainer.datasets import TransformDataset
from chainer.datasets import LabeledImageDataset
from sklearn.datasets import load_iris
from chainer.datasets import tuple_dataset
from PIL import Image
import pydicom
import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
from common.dicom_dataset1101 import DcmDataSet
from chainer.training import extensions
from chainer.backends.cuda import to_cpu
from chainer.dataset import concat_examples

gpu_flag = 0
if gpu_flag >= 0:
    cuda.check_cuda_available()

xp = cuda.cupy if gpu_flag >= 0 else np

class AbeConv(Chain):
    def __init__(self, n_out=2):
        super(AbeConv, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=None, out_channels=30, ksize=3, stride=1)
            self.conv2 = L.Convolution2D(in_channels=None, out_channels=60, ksize=3, stride=1)
            self.conv3 = L.Convolution2D(in_channels=None, out_channels=120, ksize=3, stride=1)
            self.fc4 = L.Linear(None, n_out)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2, 1)
        h = F.relu(self.conv2(x))
        h = F.max_pooling_2d(h, 2, 2,1)
        h = F.relu(self.conv3(x))
        if chainer.config.train:
            return self.fc4(h)
        return F.softmax(self.fc4(h))

if __name__ == '__main__':
    image_files_pos =glob.glob('C:/Users/K.ABE/Documents/Visicoil/training_data_102/合成画像/test/*.dcm')
    print('マーカ画像=',len(image_files_pos))
    image_files_neg = glob.glob('C:/Users/K.ABE/Documents/dataset_negative/all_negative_0703/test/*.dcm')
    print('背景画像=',len(image_files_neg))
    dataset = DcmDataSet(image_files_pos, image_files_neg)
    split_at = int(len(dataset) * 0.8)
    train_dataset, test_dataset = chainer.datasets.split_dataset(dataset, split_at)

    batchsize = 128
    train_iter = iterators.SerialIterator(train_dataset, batchsize)
    test_iter = iterators.SerialIterator(test_dataset, batchsize, False, False)
    print('train_data=', len(train_dataset))
    print('validation_data=' ,len(test_dataset))
    print("ok")

    model = AbeConv(n_out=2)

    gpu_id = 0  # Set to -1 if you use CPU
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    max_epoch = 50

    # Choose an optimizer algorithm
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)#値を変える

    # Give the optimizer a reference to the model so that it
    # can locate the model's parameters.
    optimizer.setup(model)

    input_train_path='C:/Users/K.ABE/Documents/Visicoil/training_result/1106/合成あり/unclear/モデルC/1106_train_data_final.csv'
    input_valid_path='C:/Users/K.ABE/Documents/Visicoil/training_result/1106/合成あり/unclear/モデルC/1106_valid_data_final.csv'
    input_TrainLoss_path='C:/Users/K.ABE/Documents/Visicoil/training_result/1106/合成あり/clear/モデルC/losstrain_data(modelC)_final.csv'
    input_ValidLoss_path='C:/Users/K.ABE/Documents/Visicoil/training_result/1106/合成あり/clear/モデルC/loss_valid_data(modelC)_final.csv'

    train_accuracies=[]
    train_losses=[]
    while train_iter.epoch < max_epoch:

        # ---------- One iteration of the training loop ----------

        train_batch = train_iter.next()

        x, t= concat_examples(train_batch, gpu_id)
        # Calculate the prediction of the network
        prediction_train = model(x)

        # Calculate the accuracy
        accuracy = F.accuracy(prediction_train, t)
        accuracy.to_cpu()
        train_accuracies.append(accuracy.array)

        # Calculate the loss with softmax_cross_entropy
        loss = F.softmax_cross_entropy(prediction_train,t)

        train_losses.append(to_cpu(loss.array))

        # Calculate the gradients in the network
        model.cleargrads()
        loss.backward()
        # Update all the trainable parameters
        optimizer.update()

        # --------------------- until here ---------------------


        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

            test_losses = []
            test_accuracies = []

            for test_batch in test_iter:
                image_test, target_test = concat_examples(test_batch, gpu_id)

                # Forward the test data
                prediction_test = model(image_test)

                result = F.softmax(prediction_test) #マーカ確率を算出
                print(result)
                # Calculate the loss
                loss_test = F.softmax_cross_entropy(prediction_test, target_test)
                test_losses.append(to_cpu(loss_test.array))

                # Calculate the accuracy
                accuracy = F.accuracy(prediction_test, target_test)
                accuracy.to_cpu()
                test_accuracies.append(accuracy.array)

            test_iter.reset()
            #学習したCNNモデルを保存
            serializers.save_npz('C:/Users/K.ABE/Documents/Visicoil/CNNmodel/20210106/RSSmodel/RSS_model_clear.npz', model)
