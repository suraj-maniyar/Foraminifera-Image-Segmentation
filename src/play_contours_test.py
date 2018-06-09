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

def loadData(location):
     fnames = []
     images = []
     for (dirpath, dirnames, filenames) in walk(location):
         fnames.extend(filenames)
         break

     fnames.sort()

     for f_idx in range(len(fnames)):
         mat_contents = sio.loadmat(location+'/'+fnames[f_idx])

         images.append(mat_contents['prob_map'])

     return np.asarray(images)

if __name__ == '__main__' :

    fnames2 = []

    path = 'test/ProbMap'
    X_test  = loadData(path)
    for (dirpath, dirnames, filenames) in walk(path):
        fnames2.extend(filenames)
        break
    fnames2.sort()
    # Load the images
    fnames = []
    path = 'thinned6_test'
    for (dirpath, dirnames, filenames) in walk(path):
        fnames.extend(filenames)
        break
    fnames.sort()

    for f_idx in range(len(fnames)):

        #load image and convert it to grayscale
        im =  cv2.imread('thinned6_test/'+str(fnames[f_idx]))

        gs = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

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

        sio.savemat('water_test_mat/'+fnames2[f_idx], {'label_im':labels, 'prob_map':X_test[f_idx]})
        cv2.imwrite('water_test/image_'+str(f_idx).zfill(3)+'.png',labels*20)

        #cv2.imshow('watershed',labels)
