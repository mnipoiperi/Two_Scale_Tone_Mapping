import cv2
import sys
import math
import numpy as np
from skimage import io, color

def decomposition(img, mode):
    if mode == 'L0':
    ### L0 ###
        print('QQ')

    ### Bilateral filter ###
    elif mode != 'BF':
        print('You have wrong mode! So, use BF to decompositionˋˊ')
    # Settings
    sigma_s = 30
    sigma_r = 0.2
    # decomposition
    img_base = cv2.bilateralFilter(img, 10, sigma_r, sigma_s)
    img_detail = img-img_base
    return img_base, img_detail

if __name__ == "__main__":
    ### image loading ***
    inputName = str(sys.argv[1])
    mode_decomposition = str(sys.argv[2]) #BF or L0

    ### Settings ###
    img_input_RGB = cv2.imread(inputName)
    img_input_RGB = img_input_RGB.astype(np.float32)
    #img_input_RGB = cv2.resize(img_input_RGB, (0,0), fx = 0.3, fy = 0.3)
    img_input_gray = cv2.cvtColor(img_input_RGB, cv2.COLOR_BGR2GRAY)
    img_input_gray = img_input_gray.astype(np.float32)/255

    ### Decomposition into Detail and Base layers ###
    img_base, img_detail = decomposition(img_input_gray, mode_decomposition)
    
    ### combination ###
    compressionfactor = 0.15/(np.max(img_base)-np.min(img_base))
    log_absolute_scale = np.max(img_base)*compressionfactor
    print(compressionfactor)
    print(log_absolute_scale)
    #img_out = (img_base*compressionfactor+ img_detail*1 - 0)/compressionfactor
    img_out = (img_base*compressionfactor+ img_detail*1 - 0)
    cv2.imshow("output_gray1", img_out)
    
    ### correction ###
    gamma = 1.0/3.2
    bias = -img_out.min()
    gain = 0.45
    img_out = (gain*(img_out+bias))**gamma
    img_out[img_out>1] = 1
    img_out[img_out<0] = 0
    cv2.imshow("output_gray2", img_out)

    ### restore color ###
    # LAB
    img_out_LAB = np.zeros(img_input_RGB.shape, dtype = img_input_RGB.dtype)
    img_input_LAB = cv2.cvtColor(img_input_RGB, cv2.COLOR_BGR2LAB)
    img_out_LAB[:,:,1:3] = img_input_LAB[:,:,1:3]
    img_out_LAB[:,:,0] = img_out.astype(np.uint8)
    img_out_RGB = cv2.cvtColor(img_out_LAB, cv2.COLOR_LAB2BGR)
    # RGB
    grayRatio = np.divide(img_out, (img_input_gray+0.1))
    img_out_RGB = np.zeros(img_input_RGB.shape, dtype = img_input_RGB.dtype)
    img_out_RGB[:,:, 0] = img_input_RGB[:,:,0] * grayRatio
    img_out_RGB[:,:, 1] = img_input_RGB[:,:,1] * grayRatio
    img_out_RGB[:,:, 2] = img_input_RGB[:,:,2] * grayRatio
    # change dtype
    img_out_RGB = img_out_RGB.astype(np.uint8)
    img_input_RGB = img_input_RGB.astype(np.uint8)

    ### show median products ###
    cv2.imshow("input_gray", img_input_gray)
    ### show colorful output ###
    cv2.imshow("input_RGB", img_input_RGB)
    cv2.imshow("output_RGB", img_out_RGB)    
    cv2.waitKey()

    ### output ###
    cv2.imwrite('output/' + str(sys.argv[3]) ,img_out_RGB )