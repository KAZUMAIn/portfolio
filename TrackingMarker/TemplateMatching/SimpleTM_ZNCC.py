#テンプレートマッチング
'''
Created on 2022/03/12

@author: K.ABE
'''
#from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import PIL
from PIL import Image, ImageDraw
import glob
import pydicom
from Templatematching.ExtractTemplate import load_dcm_image
from matplotlib import pyplot as plt
import time

#SAを動かすかどうか
MOVE_SA=True

#Zero means Normalized Cross Correlation
def zncc(template_array, cripped_array):
    #int型をfloat型に変換する
    tmp_ary_float=template_array.astype(np.float32)
    cp_ary_float=cripped_array.astype(np.float32)

    zncc_array_1=(tmp_ary_float - np.average(tmp_ary_float)) #分子
    zncc_array_2=(cp_ary_float-np.average(cp_ary_float))
    zncc_array_12=np.sum((zncc_array_1)* (zncc_array_2))
    zncc_array_3 =np.sum((tmp_ary_float - np.average(tmp_ary_float))*(tmp_ary_float - np.average(tmp_ary_float)))#分母
    zncc_array_4=np.sum((cp_ary_float-np.average(cp_ary_float))*(cp_ary_float-np.average(cp_ary_float)))
    zncc_array_3=np.sqrt(zncc_array_3)
    zncc_array_4=np.sqrt(zncc_array_4)
    zncc_array_34= (zncc_array_3)*( zncc_array_4)
    zncc=(zncc_array_12) /(zncc_array_34)

    return zncc
def save_dcm_image(output_array, ref_dcm, output_path):
    ref_dcm.PixelData = output_array.tobytes()
    ref_dcm.Rows, ref_dcm.Columns = output_array.shape
    ref_dcm.save_as(output_path)
    return
#template matchingを実行する
def templateMatching(template_array, input_array, sa):
    #サーチエリア内の画像を切り抜く
    sa_array=input_array[sa[1]:sa[1]+sa[3], sa[0]: sa[0]+sa[2]]
    h_c, w_c=sa_array.shape
    h_t, w_t=template_array.shape
    val_list=[]#テンプレートマッチングのスコアを保存するリスト

    #サーチエリアの中をラスタースキャンする
    for y in range(h_c-h_t):
        for x in range(w_c-w_t):
            cripped_array=sa_array[y:y+h_t, x:x+w_t]
            val=zncc(template_array, cripped_array)
            val_list.append(val)

    val_array=np.array(val_list)
#     reshaped_val_array=val_array.reshape(18,18)
#     x = np.arange(len(reshaped_val_array[0])) #x座標
#     y = np.arange(len(reshaped_val_array[1]))    #y座標

#     X, Y = np.meshgrid(x, y)
#
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     surf = ax.plot_surface(X, Y, reshaped_val_array )
    val_array=np.array(val_list)
    max_index = np.argmax(val_array)
#     print("最大",max_index)
    max_val = np.amax(val_array)

#     # ここからグラフ描画
#     # フォントの種類とサイズを設定する。
#     plt.rcParams['font.size'] = 24
#     plt.rcParams['font.family'] = 'Times New Roman'
#
#     # グラフの入れ物を用意する。
#     fig = plt.figure()
#     ax1 = Axes3D(fig)
#
#     # 軸のラベルを設定する。
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.set_zlabel('Similarity')
#
#     # データプロットする。
#     ax1.plot_surface(X, Y, reshaped_val_array,cmap='jet')
#     plt.legend()
#
#     # グラフを表示する。
#     plt.show()
#     plt.close()
    posX=max_index%(w_c-w_t)#%余りだけを計算
    posY=max_index//(w_c-w_t)#//商だけを計算
    return posX, posY

if __name__ == '__main__':
    tracking_image_list=glob.glob("C:/Users/K.ABE/Desktop/G1-1000880/20201210162955/*.dcm")
    dcm_array=load_dcm_image(tracking_image_list[0])
    template_size=32
    marker_posX=339
    marker_posY=610
    template_array = dcm_array[int(marker_posY-template_size/2):int(marker_posY+template_size/2),
                                int(marker_posX-template_size/2):int(marker_posX+template_size/2)]
    print("template_array.shape=", template_array.shape)

    #切り出したテンプレート画像を確認する。
    plt.imshow(template_array, cmap='gray')
    plt.show()

    ref_dcm= pydicom.dcmread( tracking_image_list[2] , force=True)#reference用のdicom画像を読み込む。
    template_path='C:/Users/K.ABE/Documents/Visicoil/TemplateMatching/template_image/0000.dcm'
    save_dcm_image(template_array, ref_dcm, template_path)
    print("save ok")
    #テンプレート画像のpath
    template_array=load_dcm_image(template_path)
    h_t, w_t = template_array.shape
    #追跡対象の画像のpath

#     for i in range(0,len(tracking_image_list)):
    for i in range(0,len(tracking_image_list)):
        print("ls[" + str(i) +"]=" +tracking_image_list[i])

    #追跡範囲(sa=search area)
    sa_posX=314#左上のx
    sa_posY=585#左上のy
    #sa_posX=403-25#左上のx
    #sa_posY=222-25#左上のy
#     sa_lengthX=68
#     sa_lengthY=68
    sa_lengthX=50
    sa_lengthY=50
    sa=[sa_posX, sa_posY, sa_lengthX, sa_lengthY]#めんどいのでリストにまとめる

    tracking_result_path='C:/Users/K.ABE/Documents/Visicoil/TemplateMatching/G1-1000880/position/TemplateMatching_position.csv'
    time_path='C:/Users/K.ABE/Documents/Visicoil/TemplateMatching/G1-1000880/time/TemplateMatching_time.csv'

#     with open(tracking_result_path, 'w') as f:
#         f.write('{0:<10},{1}\n'.format('markerposX', 'markerposY'))
#         f.close
#     #計算時間の内訳
#     with open(time_path, 'w') as f:
#             f.write('{0:<10},{1}\n'.format('frame', 'マーカ認識時間'))
#             f.close

    for i, path in enumerate(tracking_image_list):
#     for path in range(len(tracking_image_list)):
#         if i <-1:
        if i <1:
            continue
        print(i)
#             print("----------------------------------------------")
        t_start=time.time()

        #画像の読み込み
        tracking_image_array=load_dcm_image(path)

        posX, posY = templateMatching(template_array, tracking_image_array, sa)
        marker_posX=posX + sa[0] + template_size/2
        marker_posY =posY + sa[1] + template_size/2
        t_end=time.time()
        elapsed_time=(t_end-t_start)*1000
#         with open(time_path, 'a') as f:
#             f.write('{0:<10},{1}\n'.format(str(i), elapsed_time))
#             f.close()
#         with open(tracking_result_path, 'a') as f:
#             f.write('{0:<10},{1}\n'.format(marker_posX, marker_posY))
#         f.close()


        #経過時間を表示
        print("elapsed_time="+str(elapsed_time)+"[s]")

        #計算時間の内訳をCSVに出力していく
        #追跡画像の読み込み時間
#         with open(time_path1, 'a') as f:
#             f.write('{0:<10},{1}\n'.format(str(i+1), elapsed_time1))
#             f.close()
#         #テンプレートマッチングの時間
#         with open(time_path2, 'a') as f:
#             f.write('{0:<10},{1}\n'.format(str(i+1), elapsed_time2))
#             f.close()

        #テンプレートマッチングからマーカ位置の計算時間
#         with open(time_path3, 'a') as f:
#             f.write('{0:<10},{1}\n'.format(str(i+1), elapsed_time3))
#             f.close()

        #画像に表示してみる。
        LEVEL=3324
        WIDTH=4406
        #dicom画像は16bitなので、8bitｎ変換する。
        image_8bit = (tracking_image_array - (LEVEL - WIDTH/2))/WIDTH*255
        image_8bit[image_8bit>=255]=255
        image_8bit[image_8bit<=0]=0
#         t_end4=time.time()
#         print("time4=" + str(t_end4-t_end3) + "[s]")
        #サーチエリアと追跡位置の表示のため、PILに変換する。
        image_pil = Image.fromarray(image_8bit).convert("RGB")
        dr=ImageDraw.Draw(image_pil)
        dr.rectangle((sa[0]+posX, sa[1]+posY, sa[0]+posX+template_size, sa[1]+posY+template_size), outline=(255,0,0), width=1)#追跡結果#赤枠＃10は赤枠の長さ
        dr.rectangle((sa[0], sa[1], sa[0]+sa[2], sa[1]+sa[3]), outline=(0,0,255), width=1)#サーチエリアの表示＃青枠

        plt.imshow(image_pil)
        plt.pause(0.01)
        plt.clf()

        if MOVE_SA:
            sa[0]=int(sa[0]+posX-sa[2]/2+w_t/2)
            sa[1]=int(sa[1]+posY-sa[3]/2+h_t/2)

    pass
