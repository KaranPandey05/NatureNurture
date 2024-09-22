#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2


# In[3]:


valSet = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\karan\Downloads\Plant Data\New Plant Diseases Dataset(Augmented)\Plant Dataset\valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,

)


# In[4]:


className = valSet.class_names
print(className)


# Load Model

# In[5]:


cnnModel = tf.keras.models.load_model('trained_model.keras')


# Plot Test Image

# In[16]:


imagePath = r'C:\Users\karan\Downloads\Plant Data\New Plant Diseases Dataset(Augmented)\Plant Dataset\test\fc75404e-0b13-4827-8fde-b9b92be735cc___FAM_B.Rot 0330_flipLR.JPG'

img = cv2.imread(imagePath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()


# Predicting Input

# In[19]:


import numpy as np
image = tf.keras.preprocessing.image.load_img(imagePath, target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr]) 
predictions = cnnModel.predict(input_arr)


# In[20]:


print(predictions)


# In[21]:


result_index = np.argmax(predictions) #Return index of max element
print(result_index)


# Plotting Result

# In[23]:


model_prediction = className[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()
print(model_prediction)


# In[ ]:




