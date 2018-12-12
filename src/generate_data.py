import os
import scipy.io as sio
import cv2
import numpy as np



def loadData(location):
     fnames = []
     images = []
     labels = []
     for (dirpath, dirnames, filenames) in os.walk(location):
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
    for (dirpath, dirnames, filenames) in os.walk(location):
        fnames.extend(filenames)
        break

    fnames.sort()

    for f_idx in range(len(fnames)):
        mat_contents = sio.loadmat(location+'/'+fnames[f_idx])
        images.append(mat_contents['prob_map'])

    return np.asarray(images)


if __name__ == "__main__":
    
    os.chdir('..')

    train_path = 'data/foram_train/mat_files'
    test_path = 'data/foram_test/mat_files'
    
    X_train, Y_train = loadData(train_path)
    for i in range(len(X_train)):
        cv2.imwrite('data/foram_train/images/image/image_'+str(i).zfill(3)+'.png',X_train[i])
        cv2.imwrite('data/foram_train/images/label/label_'+str(i).zfill(3)+'.png',Y_train[i]*20)
       
    print('Train images generated successfully...')

    X_test = loadDataTest(test_path)
    for i in range(len(X_test)):
        cv2.imwrite('data/foram_test/images/image/image_'+str(i).zfill(3)+'.png',X_test[i])
        
    print('Test images generated successfully...')
 
