"""
Random Forest CLASSIFER FULL
"""

import sys
import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

def extract_feature(image_file):
    global SHAPE
    img = cv2.imread(image_file, 0)
    img = cv2.resize(img, SHAPE, interpolation = cv2.INTER_CUBIC)
    img = img.flatten()
    img = img/255
    return img
   


def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(max_depth=10)
    clf.fit(features, target)
    return clf
 


def main(directory):
   # generating two numpy arrays for features and labels
   feature_array, label_array = read_files(directory)


   print('TOTAL FEATURES shape::', feature_array.shape)
   print('TOTAL LABELS shape::', label_array.shape)
   # Splitting the data into test and training splits
   X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.2, random_state=42)

   # Train and Test dataset size details
   print ("Train_x Shape :: ", X_train.shape)
   print ("Train_y Shape :: ", y_train.shape)
   print ("Test_x Shape :: ",  X_test.shape)
   print ("Test_y Shape :: ",  y_test.shape)


   # Create random forest classifier instance
   trained_model = random_forest_classifier(X_train, y_train)
   print ("Trained model :: ", trained_model)

   predictions = trained_model.predict(X_test)
   
   for  i in range(0, 5):
       print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], predictions[i]))

   print ("Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train)))
   print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
   print (" Confusion matrix ", confusion_matrix(y_test, predictions))

   print ("Saving model...")
   pickle.dump(trained_model, open("random_forest_{}.pkl".format(str(SHAPE)), "wb"))


if __name__ == "__main__":
   if len(sys.argv) < 2:
      print ("Usage: python svm2.py ./<folder>/")
      exit()

   # Directory containing subfolders with images in them.
   directory = sys.argv[1]
   main(directory)

  