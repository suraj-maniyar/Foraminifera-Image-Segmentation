from __future__ import print_function
import numpy as np
import scipy
from scipy.misc import imread
import maxflow.fastmin
import os
import scipy.io as sio
import cv2

from generate_data import *

os.chdir('..')


if __name__ == '__main__' :

    # Read images
    path = 'data/foram_train/images/'
    
    X_train = []
    Y_train = [] 
    fname = []   
    for root, dirs, files in os.walk(path+'image'):
        for file_name in files:
            fname.append(file_name)
    fname.sort()
    
    for file_name in fname: 
        img = cv2.imread(path+'image/'+file_name)
        X_train.append(img)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    print(X_train.shape)

  
    for i in range(len(X_train)):

        img = X_train[i]

        # Create the graph.
        g = maxflow.Graph[int]()

        # Add the nodes. nodeids has the identifiers of the nodes in the grid.
        nodeids = g.add_grid_nodes(img.shape)

        # Add non-terminal edges with the same capacity.
        g.add_grid_edges(nodeids, 50)

        # Add the terminal edges. The image pixels are the capacities
        # of the edges from the source node. The inverted image pixels
        # are the capacities of the edges to the sink node.
        g.add_grid_tedges(nodeids, img, 255-img)

        # Find the maximum flow.
        g.maxflow()

        #g.maxflow.fastmin.aexpansion_grid(D, V, 10, initLabels)
        # Get the segments of the nodes in the grid.
        sgm = g.get_grid_segments(nodeids)

        # The labels should be 1 where sgm is False and 0 otherwise.
        img2 = np.int_(np.logical_not(sgm))

        cv2.imwrite('data/image_fine/train/image_'+str(i).zfill(3)+'.png',img2*255)
    print('Generated fine images successfully')
