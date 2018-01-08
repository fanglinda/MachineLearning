import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
#clf = SVC(kernel="linear")      # first##
clf = SVC(kernel="linear")

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

#### store your predictions in a list named pred

pred=clf.predict(features_test)
print ("training time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print (acc)

def submitAccuracy():
    print (acc)
    return acc