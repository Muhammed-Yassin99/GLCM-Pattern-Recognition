import random
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa
from sklearn import preprocessing
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

!wget https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip

!unzip /content/FMD.zip

paths=[]
images=[]
labels=[]
for dirname, _, filenames in os.walk('/content/image'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        img0=cv2.imread(os.path.join(dirname, filename))
        img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        images.append(img1)
        labels.append(dirname.split("/")[3])
len(paths)

indices = []
nums=[0,1,2,3,4,5,6,7,8,9]
for num in nums:
    index = random.randint(0, len(images))
    indices.append(index)

for i in indices:
    plt.title(labels[i])
    plt.imshow(images[i], cmap='gray')
    plt.show()

#Image Augmentation
augmentation = iaa.Sequential([
    # Flip
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
])

augmented_images = augmentation(images=images)
images += augmented_images
labels += labels

print(len(augmented_images))

images = np.array(images)
print("images shape:", images.shape)
print(len(labels))

for img in augmented_images[:10]:
    plt.imshow(img, cmap='gray')
    plt.show()

#Normalizing the image
#Imgs_Normed=[]
#for image in images:
  #norm_img=cv2.normalize(image, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
  #norm_img = np.zeros((800,800))
  #final_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
  #Imgs_Normed.append(norm_img)

#len(Imgs_Normed)

#Encode labels from text to integers
images = np.array(images)
labels = np.array(labels)
label_encoder = preprocessing.LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

print(np.unique(labels))
print(np.unique(encoded_labels))

x_train, x_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.3, random_state=42)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

def Get_GLCM(data):
    glcm_features = []
    for img in data:
        glcm = greycomatrix(img, distances=[5], angles=[0])
        energy = greycoprops(glcm, 'energy')
        contrast = greycoprops(glcm, 'contrast')
        dissimilarity = greycoprops(glcm, 'dissimilarity')
        homogeneity = greycoprops(glcm, 'homogeneity')
        correlation= greycoprops(glcm,'correlation')
        ASM= greycoprops(glcm,'ASM')
        #shade= greycoprops(glcm,'shade')---> it gives an error that shade is invalid property
        feature = np.array([energy, contrast, dissimilarity, homogeneity, correlation, ASM]).flatten()
        glcm_features.append(feature)
    return glcm_features

glcm_train = Get_GLCM(x_train)
glcm_train = np.array(glcm_train)
print("GLCM_train shape:", glcm_train.shape)

glcm_test = Get_GLCM(x_test)
glcm_test = np.array(glcm_test)
print("GLCM_test shape:", glcm_test.shape)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(glcm_train, y_train)
predictions = knn.predict(glcm_test)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy Score =', accuracy * 100, '%')
