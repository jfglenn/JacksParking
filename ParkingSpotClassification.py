# %% [markdown]
# # 1. Setup And Load Data 
# ##### 1.1 Install Dependencies & Import 

# %%
# I installed the packages below through the terminal
# !pip install tensorflow tensorflow-gpu opencv-python matplotlib 

import tensorflow as tf
import os

# %%
# Avoid Out Of Memory errors by setting GPU memory consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# %% [markdown]
# ##### 1.2 Load Data

# %%
import cv2 # opencv
import imghdr # check file extentions for certain images

# %%
image_extns = ['jpeg', 'jpg', 'png']
data_dir = 'data'

# %%
#img=cv2.imread(os.path.join('data', 'available', 'Screenshot 2023-03-17 153704.png'))
#img.shape
#plt.imshow(img)

# %%
# Check every image within the main 'Data' folder for incorrect
# file extention types
for folder in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, folder)):
        image_path = os.path.join(data_dir, folder, image)
        try:
            #img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_extns:
                print('Image does not have valid extention', format(image_path))
                os.remove(image_path)
        except:
                print('Could not read image', format(image_path))

# %%
tf.data.Dataset

# %%
import numpy as np 
from matplotlib import pyplot as plt

# %%
# Builds image data set (this is the data pipeline)
data = tf.keras.utils.image_dataset_from_directory(data_dir)

# %%
# Data set is not preloaded into memory already (it is a 
# only a generator), so we have to convert data set into
# numpy iterator to access data 
data_iterator = data.as_numpy_iterator()

# %%
# Get batch of data
batch = data_iterator.next()

# %%
# images represented as numpy arrays
batch[0].shape

# %%
batch[1].shape

# %%
# Check what each flag is
# folder 0 = available
# folder 1 = unavailable
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# %% [markdown]
# # 2. Preprocess Data
# ##### 2.1 Scale Data

# %%
scaledData = data.map(lambda x,y: (x/255, y))

# %%
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# %% [markdown]
# ##### 2.2 Categorize Data 
# ###### into training and testing partition 

# %%
len(scaledData) #14 batches


# %%
# training data: what is used to train deep learning model
# validation data: used to evaluate model while training (fine tuning)
# test data: used post training to evaluate training
train_size = int(len(scaledData)*.7)
val_size = int(len(scaledData)*.2)+1
test_size = int(len(scaledData)*.1)+1

# %%
# take: defines how much data to use for partition 
train = scaledData.take(train_size)
validate = scaledData.skip(train_size).take(val_size)
test = scaledData.skip(train_size + val_size).take(test_size)

# %% [markdown]
# # 3. Deep Model
# ##### 3.1 Build Deep Learning Model 

# %%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# %%
# Establishing Sequential Class
model = Sequential()

# %%
# Adding convolutional layer and Mask Pool Layer
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# %%
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# %%
model.summary()

# %% [markdown]
# ##### 3.2 Train

# %%
logdir = 'logs'

# %%
# Logs how the model performs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# %%
# .fit takes in training data
history = model.fit(train, epochs=20, validation_data=validate, callbacks=[tensorboard_callback])

# %%
history.history

# %% [markdown]
# ##### 3.3 Plot Data

# %%
# show loss
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# %%
# show accuracy
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# %% [markdown]
# # 4. Evaluate Performance
# ##### 4.1 Evaluate

# %%
from keras.metrics import Precision, Recall, BinaryAccuracy

# %%
# 
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

# %%
# Loop through each batch in testing data
# x, y is set of images
# 
for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)

# %%
print(f'Precision:{precision.result().numpy()}, Recall:{recall.result().numpy()}, Accuracy:{accuracy.result().numpy()}')

# %% [markdown]
# ##### 4.2 Test

# %%
img = cv2.imread('unavailable_test.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# %%
# format image for testing
resizeImg = tf.image.resize(img, (256, 256))
plt.imshow(resizeImg.numpy().astype(int))
plt.show()

# %%
yhat = model.predict(np.expand_dims(resizeImg/255,0))

# %%
yhat

# %%
if yhat > 0.5:
    print(f'Predicted class is uavailable')
else:
    print(f'Predicted class is available')

# %% [markdown]
# # 5. Save The Model
# ##### 5.1 Save The Model

# %%
from keras.models import load_model

# %%
model.save(os.path.join('models', 'ParkingSpotStatusModel.h5'))

# %%
new_model = load_model(os.path.join('models', 'ParkingSpotStatusModel.h5'))

# %% [markdown]
# # 6. Reuseable Function 

# %%
def SpotStatus(parkingSpots, model):
    availableSpots = len(parkingSpots)
    #parkingSpotStatusModel = load_model(os.path.join('models', 'ParkingSpotStatusModel.h5'))
    for spot in parkingSpots:
        model.predict(np.expand_dims(spot.x/255,0))
        if yhat > 0.5:
            availableSpots = availableSpots - 1
    return availableSpots



# %%
