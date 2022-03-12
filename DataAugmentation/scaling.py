#画像の拡大・縮小
'''
Created on 2022/03/12

@author: K.ABE
'''
import cv2
import numpy as np
import pydicom
import glob
import matplotlib.pyplot as plt
import random
def load_dcm_image(path=None):
    dcm = pydicom.dcmread(path , force=True)#dicom画像を読み込む。
    dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    return dcm.pixel_array

def save_dcm_image(output_array, ref_dcm, output_path):
    ref_dcm.PixelData = output_array.tobytes()
    ref_dcm.Rows, ref_dcm.Columns = output_array.shape
    ref_dcm.save_as(output_path)
    return

if __name__ == '__main__':
    #読み込む画像
    image_files_pos_upsize=glob.glob('C:/Users/K.ABE/Documents/Visicoil/training_data_1027/100pixel/rotation(clear)/rotated_img1_0000.dcm')
    ref_path='C:/Users/K.ABE/Documents/dataset_positive/CoreView_12_FPD_CP1/0001.dcm'

    #dicom画像の読み込み
    print(len(image_files_pos_upsize))
    for i in range(0, len(image_files_pos_upsize)):
        mk_img=load_dcm_image(image_files_pos_upsize[i])
        ref_dcm= pydicom.dcmread( image_files_pos_upsize[0] , force=True)#reference用のdicom画像を読み込む。
        w,h=mk_img.shape
        #拡大・縮小の倍率をランダムに決める
        x=random.uniform(0.8, 1.2 )
        #読み込んだ画像をランダムの拡大率と縮小率に設定
        re_w=random.uniform(w*x, w*x)
        re_h=random.uniform(h*x, h*x)
        #マーカ画像をリサイズ
        resize_mk_img=cv2.resize(mk_img, dsize=(int(re_h),int(re_w)))
        plt.imshow(resize_mk_img, cmap='gray')
        plt.show()
        w2,h2=resize_mk_img.shape
        marker_posX=w2/2
        marker_posY=h2/2

        image_size=32

        #26×26に切り取る
        image_array = resize_mk_img[int(marker_posY-image_size/2):int(marker_posY+image_size/2),
                                    int(marker_posX-image_size/2):int(marker_posX+image_size/2)]

        save_path='C:/Users/K.ABE/Documents/Visicoil/training_data_1027/100pixel/sample/scaled_image1.2_'+str(i).zfill(4)+'.dcm'
        scaled_image=save_dcm_image(image_array, ref_dcm, save_path)
    print('save ok')
