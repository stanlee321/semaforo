"""
SVM CLASSIFIER

"""

import sys
import os
import cv2
import numpy as np

import pickle

SHAPE = (8, 24)

def extract_feature(image_file):
   #print(image_file)
   img = cv2.imread(image_file)
   #img = cv2.resize(img, (200,200))
   #cv2.imwrite('this_1.jpg',img)
   img = cv2.resize(img, SHAPE, interpolation = cv2.INTER_CUBIC)
   img = img.flatten()
   #print(img.shape)
   img = img/(np.mean(img)+0.0001)
   return img


if __name__ == "__main__":
   pic = '1.png'
   feature_img = extract_feature(pic)
   X_test = np.asarray(feature_img)

   print( 'checking for model....')
   if os.path.isfile("svm_model_{}.pkl".format(str(SHAPE))):
      print ("Using previous model...")
      svm = pickle.load(open("svm_model_{}.pkl".format(str(SHAPE)), "rb"))
   else:
      print ("Retrain DUde")
   print ("Testing...\n")
  
   x = X_test
   x = x.reshape(1, -1)
   prediction = svm.predict(x)[0]

   print(prediction)