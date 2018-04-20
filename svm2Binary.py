"""
SVM CLASSIFER FULL
"""

import sys
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import time
from sklearn import svm as svc

#import cPickle as pickle
import pickle

SHAPE = (8,24)

def read_files(directory):
   print ("Reading files...")
   s = 1
   feature_list = list()
   label_list   = list()
   num_classes = 0
   for root, dirs, files in os.walk(directory):
      for d in dirs:
         num_classes += 1
         images = os.listdir(root+d)
         for image in images:
            s += 1
            label_list.append(d)
            feature_list.append(extract_feature(root+d+"/"+image))

   print (str(num_classes) + " classes")
   return np.asarray(feature_list), np.asarray(label_list)

#counter = 0
def extract_feature(image_file):

   #global counter
   #print(image_file)
   img = cv2.imread(image_file, 0)

   #filter_g = cv2.bilateralFilter(img,35,75,75)
   filter_g = cv2.GaussianBlur(img,(15,15),0)
   full_mask   = filter_g
   
   img = cv2.resize(full_mask, SHAPE, interpolation = cv2.INTER_CUBIC)
   #cv2.imwrite('./tests/{}.jpg'.format(counter), img)
   img = img.flatten()
   #counter +=1
   #print('SHape is', img.shape)

   #img = img/(np.mean(img)+0.0001)
   img = img/255
   return img
 


def buildSVM(features, target):
   print ("Fitting")
   C = 0.1    # SVM regularization parameter
   #C = 10
   #svm = SVC()
   #svm = SVC(kernel='linear', C = C)
   #svm = SVC(kernel='rbf', gamma=0.7, C=C)  # RBF
   #svm = SVC(kernel='poly', degree=2, C=C)
   svm = svc.LinearSVC(C=C)
   # Fitting model
   svm.fit(features, target)

   return svm



def main(directory):
   # generating two numpy arrays for features and labels
   feature_array, label_array = read_files(directory)
   # Splitting the data into test and training splits
   print('TOTAL FEATURES shape::', feature_array.shape)
   print('TOTAL LABELS shape::', label_array.shape)
   # Splitting the data into test and training splits
   X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.2, random_state=42)

   # Train and Test dataset size details
   print ("Train_x Shape :: ", X_train.shape)
   print ("Train_y Shape :: ", y_train.shape)
   print ("Test_x Shape :: ",  X_test.shape)
   print ("Test_y Shape :: ",  y_test.shape)

   svm = buildSVM(X_train, y_train)

  
   print ("Saving model...")
   pickle.dump(svm, open("svm_LINEARSVC.pkl", "wb"))

   print ("Trained model :: ", svm)

   print ("Testing...\n")
  
   right = 0
   total = 0

   for x, y in zip(X_test, y_test):
      x = x.reshape(1, -1)
      prediction = svm.predict(x)[0]
      if y == prediction:
         right += 1
      total += 1

   accuracy = float(right)/float(total)*100

   print ("Accuracy:: ",  str(accuracy) + "% accuracy")


   predictions = svm.predict(X_test)
   
   for  i in range(0, 5):
       print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], predictions[i]))

   print ("Train Accuracy :: ", accuracy_score(y_train, svm.predict(X_train)))
   print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
   print (" Confusion matrix ", confusion_matrix(y_test, predictions))

   print ("Saving model...")

if __name__ == "__main__":
   if len(sys.argv) < 2:
      print ("Usage: python svm2.py ./<folder>/")
      exit()

   # Directory containing subfolders with images in them.
   directory = sys.argv[1]

   main(directory)