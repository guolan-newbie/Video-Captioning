#-*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import pandas as pd
import skimage  #将图片作为numpy数组处理
from cnn_util import *


def preprocess_frame(image, target_height=224, target_width=224):


    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    #print(str(height)+" & "+str(width)+" & "+str(rgb))
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]
    #print(resized_image)
    return cv2.resize(resized_image, (target_height, target_width))
    #图片尺寸标准化

def main():
    num_frames = 80  ##Default : Max Num of Frames is 80
    
    ### Pretrained Caffe Model ###
    vgg_model = './VGG_ILSVRC_16_layers.caffemodel'
    vgg_deploy = './VGG16_deploy.prototxt'
    #################################

    ### vedio source path ###
    video_path = '../train-video'
    #################################

    ### video save path ###
    video_save_path = './rgb_feats'
    ################################
    
    ### get videos list ###
    videos = os.listdir(video_path)
    
    videos = filter(lambda x: x.endswith('mp4'), videos)
    
    
    #################################

    ### build model ###
    cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)
    #################################
    #input("pause")
    ### Enumerate Videos ###
    for idx, video in enumerate(videos):
        print(idx, video)

        ### Ignore the processed videos ###
        if os.path.exists( os.path.join(video_save_path, video) ): 
            print("Already processed ... ")
            continue

        ### Get Path ###
        video_fullpath = os.path.join(video_path, video)    

        try:
            ### Build VideoCapture Object ###
            cap  = cv2.VideoCapture( video_fullpath )  ## Open Video
        except:
            pass

        frame_count = 0
        frame_list = []

        while True:
            ret, frame = cap.read()  ## Read Frame
            if ret is False:  ## If there is no more frame, End.
                break

            frame_list.append(frame)  ## Append Frame
            frame_count += 1

        frame_list = np.array(frame_list)  
        #print(frame_count)
        #input("pause")
        if frame_count > 80:    ## Num of Frames > 80 , Sample 80 Frames Uniformly.
            frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int) 
            frame_list = frame_list[frame_indices]  

        cropped_frame_list = np.array(list(map(lambda x: preprocess_frame(x), frame_list)))
 
        ## map(function , list) : return function(list)
        ## process all frames
        print(len(cropped_frame_list))
        #input("pause")
        feats = cnn.get_features(cropped_frame_list)
        ## Use Pretrained Model to Process Frames..

        save_full_path = os.path.join(video_save_path, video + '.npy')
        ## Save features.
        np.save(save_full_path, feats)


if __name__ == "__main__" :
    main()
