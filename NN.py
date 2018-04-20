"""
Random Forest CLASSIFER FULL
"""

import sys
import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
    filter_g = cv2.bilateralFilter(img,35,75,75)
    filter_g = cv2.GaussianBlur(filter_g,(15,15),0)
    full_mask   = filter_g
    
    img = cv2.resize(full_mask, SHAPE, interpolation = cv2.INTER_CUBIC)
    img = img.flatten()
    img = img/255
    return img
   


def NN_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    #mlp = MLPClassifier(hidden_layer_sizes=(300, ), max_iter=40, alpha=1e-5,
    #                solver='sgd', verbose=10, tol=1e-5, random_state=1,
    #                learning_rate_init=.1)

    # activations = {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}

    mlp = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(256,), learning_rate='constant',
       learning_rate_init=0.001, max_iter = 200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
    
    mlp.fit(features, target)
    return mlp
 


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
   trained_model = NN_classifier(X_train, y_train)
   print ("Trained model :: ", trained_model)

   predictions = trained_model.predict(X_test)
   
   for  i in range(0, 5):
       print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], predictions[i]))

   print ("Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train)))
   print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
   print (" Confusion matrix ::  ", confusion_matrix(y_test, predictions))

   print("Classification Report :: ", classification_report(y_test,predictions))
   print ("Saving model...")
   pickle.dump(trained_model, open("NN_bw{}.pkl".format(str(SHAPE)), "wb"))


if __name__ == "__main__":
   if len(sys.argv) < 2:
      print ("Usage: python svm2.py ./<folder>/")
      exit()

   # Directory containing subfolders with images in them.
   directory = sys.argv[1]
   main(directory)

  