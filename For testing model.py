#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
import numpy as np


# In[ ]:


### CHANGE FILE PATH IN THIS CHUNK AND RUN THEM ALL ###
# Load the model from the HDF5 file
best_IncV3 = load_model("/path/to/load/Model.h5")
best_Xcep = load_model("/path/to/load/Model.h5")
best_ResNet50 = load_model("/path/to/load/Model.h5")
best_ResNet101 = load_model("/path/to/load/Model.h5")


#To prepare the test set, please make sure to label each image file with its respective class. 
#For instance, change the filename of each image as 'typeALS_OldName'. Use an underscore between 'typeALS' and 'OldName'. 
#For example, '3CH_image1.jpg'. 
#The image file can have different file extensions such as '.jpg', '.jpeg', '.png', '.gif', '.bmp', or '.tiff'.

path_TESTSET = "/path/to/load/image_fie/or/image_folder"


# In[ ]:


def prepare_testset(directory, label_mapping):
    # Function to load images using PIL
    def load_images_from_directory_PIL(directory):
        image_list = []
        if not os.path.exists(directory):
            print(f"The directory '{directory}' does not exist.")
            return image_list
        
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('JPG', 'JPEG', '.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff'))]
        
        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            try:
                img = Image.open(image_path)
                image_list.append(img)
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
        return image_list
    
    # Function to convert PIL images to pixel arrays
    def convert_images_to_arrays(image_list):
        pixel_arrays = []
        for img in image_list:
            img_array = np.array(img)
            pixel_arrays.append(img_array)
        return pixel_arrays
    
    # Function to extract filename from image paths
    def convert_images_to_filename(image_list):
        filenames = []
        for img in image_list:
            filename = os.path.basename(img.filename)
            filenames.append(filename)
        return filenames
    
    # Function to convert labels to one-hot encoding
    def convert_labels_to_one_hot(label_list, label_mapping):
        numerical_labels = [label_mapping[label] for label in label_list]
        num_classes = len(label_mapping)
        one_hot_labels = to_categorical(numerical_labels, num_classes=num_classes)
        return one_hot_labels
    
    # Load images using PIL
    list_pillow_TESTSET = load_images_from_directory_PIL(directory)
    
    # Convert the list of PIL images to pixel arrays
    pixel_arrays_TESTSET = convert_images_to_arrays(list_pillow_TESTSET)
    
    # Get the names of the images
    names_TESTSET = convert_images_to_filename(list_pillow_TESTSET)
    
    # Extract labels from filenames
    ALStype_TESTSET = [name.split('_')[0] for name in names_TESTSET]
    
    # Update the labels based on the mapping
    ALStype_Num_TESTSET = [label_mapping[label] for label in ALStype_TESTSET]
    
    # Create one-hot encoded labels
    one_hot_labels_TESTSET = convert_labels_to_one_hot(ALStype_TESTSET, label_mapping)
    
    # Build the dictionary
    dict_TESTSET = {'Name': names_TESTSET, 'ALStype': ALStype_TESTSET, 'PixelArrays': pixel_arrays_TESTSET, 'ALStype_Num': ALStype_Num_TESTSET, 'one_hot_labels': one_hot_labels_TESTSET}
    
    return dict_TESTSET



# In[ ]:


label_mapping = {
    "3CH": 0,
    "3CTL": 1,
    "3CN": 2,
    "3CL": 3,
    "4C": 4,
    "4CL": 5,
    "4CTL": 6
}

dict_TESTSET = prepare_testset(path_TESTSET, label_mapping)

# Variables for testing the model
X_test = np.array(dict_TESTSET['PixelArrays'])
y_test = np.array(dict_TESTSET['one_hot_labels'])
y_test_ALStype_Num = np.array(dict_TESTSET['ALStype_Num'])

# Print the one-hot encoded labels
print(dict_TESTSET["one_hot_labels"])


def test_model(loaded_best_model, X_test, y_test):
    # Assuming you have a dataset (X_test) and its corresponding true labels (y_test)
    y_pred = loaded_best_model.predict(X_test)

    # Assuming y_true is one-hot encoded, you can obtain the predicted class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)  # Assuming y_true is one-hot encoded

    # Compute Precision, Recall, and F1-score
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1_score_val = f1_score(y_true_classes, y_pred_classes, average='weighted')

    # Compute ROC-AUC for each class
    roc_auc = roc_auc_score(y_test, y_pred, average='macro')  # Assuming you have multi-class labels

    # Print the classification report which includes precision, recall, and f1-score
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

    # Print Precision, Recall, and F1-score
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1_score_val)

    # Print ROC-AUC score
    print("ROC-AUC (Macro):", roc_auc)
    


# In[ ]:


print(test_model(best_IncV3, X_test, y_test)) 
print(test_model(best_Xcep, X_test, y_test)) 
print(test_model(best_ResNet50, X_test, y_test)) 
print(test_model(best_ResNet101, X_test, y_test)) 

