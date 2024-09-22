#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Training Image Preprocessing

# In[8]:


traningSet = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\karan\Downloads\Plant Data\New Plant Diseases Dataset(Augmented)\Plant Dataset\train',
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


# Validation Image Processing

# In[9]:


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


# Building CNN

# In[11]:


cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape = [128,128,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3,  activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[12]:


cnn.add(tf.keras.layers.Dropout(0.25))


# In[13]:


cnn.add(tf.keras.layers.Flatten())


# In[14]:


cnn.add(tf.keras.layers.Dense(units=1500, activation='relu'))


# In[15]:


cnn.add(tf.keras.layers.Dropout(0.4))


# In[16]:


cnn.add(tf.keras.layers.Dense(units=38, activation='softmax'))


# Compile and Train

# In[17]:


cnn.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[18]:


cnn.summary()


# In[24]:


cnn_fit = cnn.fit(x = traningSet, 
                  validation_data = valSet, 
                  epochs = 10)


# In[23]:


cnn.save('trained_model.keras')


# In[ ]:




