from __future__ import print_function
import numpy as np
import scipy
from scipy.misc import imread
import maxflow.fastmin
from os import walk
import scipy.io as sio

import cv2

def loadData(location):
     fnames = []
     images = []
     labels = []
     for (dirpath, dirnames, filenames) in walk(location):
         fnames.extend(filenames)
         break

     fnames.sort()

     for f_idx in range(len(fnames)):
         mat_contents = sio.loadmat(location+'/'+fnames[f_idx])

         images.append(mat_contents['prob_map'])
         labels.append(mat_contents['label_im'])

     return np.asarray(images),np.asarray(labels)

def loadDataTest(location):
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

    # Read images
    path = 'test/ProbMap'
    #X_train, Y_train = loadData(path)
    X_train = loadDataTest(path)



    for i in range(len(X_train)):
        cv2.imwrite('orig_test/image_'+str(i).zfill(3)+'.png',X_train[i])
        img = imread('orig_test/image_'+str(i).zfill(3)+'.png')


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

        cv2.imwrite('fine_50_test/img_fine'+str(i).zfill(3)+'.png',img2*255)
