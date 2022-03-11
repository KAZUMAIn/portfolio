
'''
Created on 2020/07/09

@author: K.ABE
'''

import matplotlib.pyplot as plt
import cv2
import glob
import chainer
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import TransformDataset
from chainer.datasets import LabeledImageDataset
from chainer.datasets import tuple_dataset
from PIL import Image, ImageDraw
import pydicom
# import select_random
import matplotlib.pyplot as plt
from common.CNN3_Train_and_Validation import  AbeConv
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize
import time

# save_marker_likelihood_path='C:/Users/K.ABE/Documents/Visicoil/tracking_result/marker_likelihood_map/testmap.png'

#SAを動かすかどうか
MOVE_SA=True

#サーチエリアを設定
sa_posX= 314#1frame目
sa_posY= 585#1frame目
# sa_posX=376 #1frame目
# sa_posY=193 #1frame目
# sa_posX=354#左上のx149frsme
# sa_posY=172#左上のy
# sa_posX=380
# sa_posY=211
sa_lengthX=50
sa_lengthY=50
sa=[sa_posX, sa_posY, sa_lengthX, sa_lengthY]#めんどいのでリストにまとめる

xp = cuda.cupy

def zscore( x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True,dtype=xp.float32)
    xstd  = xp.std(x, axis=axis, keepdims=True,dtype=xp.float32)
    zscore = (x-xmean)/xstd
    return zscore

def load_dcm_image(path=None):
    dcm = pydicom.dcmread(path , force=True)#dicom画像を読み込む。
    dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian#おまじない。たぶんないと動かない
    return dcm.pixel_array#pixel_arrayはndarray型の画像

def classfication (path, model):
#     timeit.timeit(stmt=func1, setup=func2, timer=time.time.perf_counter, number=1000000)
    marker_likelihood_list=[]#マーカ確率を保存するリスト

    print(path)
    tracking_image=load_dcm_image(path)

    #切り出したい(Classify)画像のサイズ[pixel](偶数のみ)
    image_size=32
    step_size=1

    pos_list=[]
    neg_list=[]

    #マーカ確率を算出するプログラム

    classify_array=xp.zeros((1,1,image_size,image_size), xp.float32)
    for y in range(sa_lengthY-image_size):#
        for x in range(sa_lengthX-image_size):
            if not (y%step_size==0 and x%step_size==0):
                continue

            #サーチエリアの中で画像を切り出す
            seracharea_array = tracking_image[int(sa[1]+y):int(sa[1]+y+image_size),int(sa[0]+x):int(sa[0] +x+image_size)]

            h,w = seracharea_array.shape
            #切り出した画像を正規化
            normalized_array=zscore(xp.reshape(xp.asarray(seracharea_array, xp.float32), (1, 1,h,w))) #(時間？,チャンネル数,高さ,横)
#             t_end10=time.time()
            classify_array=xp.concatenate((classify_array,normalized_array),axis=0)

    gpu_device = 0
    cuda.get_device(gpu_device).use()
    model.to_gpu(gpu_device)


    prediction_test =model(classify_array[1:])
#     result = F.softmax(prediction_test) #マーカ確率を算出
#     print(result)
    #マーカ尤度を算出
    result = prediction_test
    marker_likelihood=result.data[:,1]
#     print(marker_likelihood)
#             pos_list.append(result.data[0][0])
#             neg_list.append(result.data[0][1])

    marker_likelihood_array=xp.asnumpy(marker_likelihood)
    marker_image_array=marker_likelihood_array.reshape(18,18)


    max_array=np.max(marker_image_array)
    min_array=np.min(marker_image_array)
    Threshold=0.7
    marker_image_array[marker_image_array<0]=0

    marker_image_array[marker_image_array <=((max_array-min_array)*float(Threshold)+min_array)] = 0
    marker_map_image=marker_image_array.astype(np.int16)
    mp_h,mp_w=marker_map_image.shape
#     print(mp_h,mp_w)
#     print('マーカ確率マップ')
#     plt.imshow(marker_map_image,cmap="jet")
#     plt.colorbar()
#     plt.imshow(marker_map_image, cmap='gray')
#     plt.show()

    mu = cv2.moments(marker_image_array, False)
    posX,posY= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
#     t_end4=time.time()
#     elapsed_time4=(t_end4-t_start4)*1000

#     crop_ary = tracking_image[int(sa[1]+(sa[3]-image_size)/2):int(sa[1]+(sa[3]-image_size)/2+image_size),int(sa[0]+(sa[2]-image_size)/2):int(sa[0]+(sa[2]-image_size)/2+image_size)]
    crop_ary = tracking_image[int(sa[1]+(sa[3]-mp_h)/2):int(sa[1]+(sa[3]-mp_h)/2+mp_h),int(sa[0]+(sa[2]-mp_w)/2):int(sa[0]+(sa[2]-mp_w)/2+mp_w)]
#     print('切り出すマーカ画像')
#     plt.imshow(crop_ary, cmap='gray')
#     plt.show()
    h, w=crop_ary.shape
    max=np.max(crop_ary)
    min=np.min(crop_ary)
    crop_ary=(crop_ary-min)/(max-min)
    crop_img=Image.fromarray(crop_ary*255).convert('RGB')
    image_img=Image.fromarray(marker_image_array).convert('RGB')
    image_img=image_img.resize((w, h))
    im = Image.blend(image_img, crop_img, 0.7)
#     print('切り出すマーカ画像')

#     plt.imshow(im)
#     plt.show()
#     マーカ尤度マップとマーカ画像を合成
#     plotdosemap(im, marker_map_image, 0, 300 , "marker likelihood", -1000, 20, save_marker_likelihood_path)
#     print("マーカ確率マップを表示")
#     plotdosemap(im, marker_map_image, 0, 300 , "marker probability", -1000, 20)
    return posX,posY


def plotdosemap(ctimage, doseimage, vmin, vmax, label, thre_ct, thre_dose):    #マーカ画像とヒートマップを合成
    fig, ax = plt.subplots()
    ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
    ax.tick_params(labelleft="off",left="off") # y軸の削除
    ax.set_xticklabels([])
    ctplot = ax.imshow(ctimage, cmap="Greys_r", norm = Normalize(vmin=500, vmax=1500))
    gammaplot = ax.imshow((doseimage), cmap="jet", alpha=0.2)

    pp = fig.colorbar(gammaplot)
    pp.ax.tick_params(labelsize=24)
    pp.set_label(label, fontname="Times New Roman", fontsize=24)
#     print('ヒートマップと認識したマーカを合成')
#     plt.savefig(fn_save, dpi=48)
    plt.show()
    return 0

if __name__ == '__main__':

    #切り出したい(Classify)画像のサイズ[pixel](偶数のみ)
    image_size=32
    step_size=1

    path="C:/Users/K.ABE/Desktop/G1-1000880/20201210162955/*.dcm"
    ls=glob.glob(path)

    #modelの読み込み
    model = AbeConv(n_out=2)

    # Load the saved parameters into the instance
#     serializers.load_npz('C:/Users/K.ABE/Documents/Visicoil/CNNmodel/20210106/RSSmodel/RSS_model_clear.npz', model)
#     serializers.load_npz('C:/Users/K.ABE/Documents/Visicoil/CNNmodel/1106/clear/RSモデル/modelA_clear.npz', model)
    serializers.load_npz('C:/Users/K.ABE/Documents/Visicoil/CNNmodel/20210106/RSSmodel/RSS_model_clear.npz', model)

#     serializers.load_npz('C:/Users/K.ABE/Documents/Visicoil/CNNmodel/1106/clear/モデルC/modelC_clear.npz', model)
#     output_path='C:/Users/K.ABE/Documents/Visicoil/tracking_result/1110/CoreView_63_FPD_CP1/clear/RSSモデル/position/RSSmodel_final_sample.csv'
#     output_path='C:/Users/K.ABE/Documents/Visicoil/tracking_result/G1-1000880/position/R-Smodel/R-Smodel_position.csv'
    output_path='C:/Users/K.ABE/Documents/Visicoil/tracking_result/G1-1000880/position/R-S-Smodel/R-S-Smodel_position.csv'
#     output_path='C:/Users/K.ABE/Documents/Visicoil/tracking_result/1110/CoreView_63_FPD_CP1(From150Frame)/clear/モデルC/position/marker_positionModelCclear.csv'
#     time_path_re='C:/Users/K.ABE/Documents/Visicoil/tracking_result/1110/CoreView_63_FPD_CP1(From150Frame)/clear/モデルC/time/RecognizeTimeModelCclear.csv'
#     time_path_re='C:/Users/K.ABE/Documents/Visicoil/tracking_result/1110/CoreView_63_FPD_CP1/clear/RSモデル/time/RSSmodel_final_sample.csv'
    time_path_re='C:/Users/K.ABE/Documents/Visicoil/tracking_result/G1-1000880/time/R-S-Smodel/R-S-Smodel_time.csv'
    #残っているファイルを殺す（マーカ位置）
#     with open(output_path,  'w') as f:
#         f.write('{0:<10},{1}\n'.format('markerposX', 'markerposY'))
#         f.close()
#     with open(time_path_re,  'w') as f:
#         f.write('{0:<10},{1}\n'.format('frame', 'マーカ認識にかかる時間'))
#         f.close()

    for i in range(0, len(ls)):
        if i<-1:
            continue

        #計算開始
        t_start = time.time()

        posX, posY=classfication(ls[i], model)
        marker_posX, marker_posY=posX+sa[0]+image_size/2, posY+sa[1]+image_size/2
        print(marker_posX, marker_posY)
        t_end=time.time()
        elapsed_time=(t_end-t_start)*1000
        print("elapsed_time",elapsed_time)

#         with open(output_path,  'a') as f:
#             f.write('{0:<10},{1}\n'.format(marker_posX, marker_posY))
#             f.close()
#         with open(time_path_re,  'a') as f:
#             f.write('{0:<10},{1}\n'.format(str(i+1),elapsed_time))
#             f.close()
        #追跡の様子を画像に表示
        LEVEL=3357
        WIDTH=4267

        #dicom画像は16bitなので、8bitに変換する。
        tracking_image_array=load_dcm_image(ls[i])
        image_8bit = (tracking_image_array - (LEVEL - WIDTH/2))/WIDTH*255
        image_8bit[image_8bit>=255]=255
        image_8bit[image_8bit<=0]=0


        LEVEL=np.argmax(tracking_image_array)/2
        WIDTH=np.argmax(tracking_image_array)-np.argmin(tracking_image_array)
        #サーチエリアと追跡位置の表示のため、PILに変換する。
        image_pil = Image.fromarray(image_8bit).convert("RGB")
        dr=ImageDraw.Draw(image_pil)

        #sa=[sa_posX, sa_posY, sa_lengthX, sa_lengthY]
        dr.rectangle((marker_posX-image_size/2, marker_posY-image_size/2,marker_posX+image_size/2, marker_posY+image_size/2), outline=(255,0,0), width=1)#追跡i位置の表示#赤枠＃10は赤枠の長さ
        dr.rectangle((sa[0], sa[1], sa[0]+sa[2], sa[1]+sa[3]), outline=(0,0,255), width=1)#サーチエリアの表示＃青枠

        plt.imshow(image_pil)
        plt.pause(0.001)
        plt.clf()
#
        sa[0]=int(marker_posX-sa[2]/2)
        sa[1]=int(marker_posY-sa[3]/2)

    pass