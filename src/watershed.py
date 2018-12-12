from __future__ import print_function
import cv2
import numpy as np
import scipy.io as sio
from os import walk
import sys
import os
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage


if __name__ == '__main__' :
    
    os.chdir('..')
    
    fnames = []
    path = 'data/image_thin/train'
    
    for (dirpath, dirnames, filenames) in walk(path):
        fnames.extend(filenames)
        break
    fnames.sort()

    for f_idx in range(len(fnames)):
        img = cv2.imread(path+'/'+str(fnames[f_idx]))
        gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Threshold and find contours
        ret,thresh1 = cv2.threshold(gs,10,255,cv2.THRESH_BINARY_INV)
        #im2, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #print(thresh1.shape)
        #thresh1 = cv2.bitwise_not(thresh1)

        # Distance transform and set segment number
        D = ndimage.distance_transform_edt(thresh1)
        localMax = peak_local_max(D, indices=False, min_distance=10,labels=thresh1)

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh1)

        cv2.imwrite('data/image_segment/train/image_'+str(f_idx).zfill(3)+'.png',labels*20)
        
