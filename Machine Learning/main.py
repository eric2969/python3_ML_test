from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
def imshow(img):
    plt.imshow(img)
batch_size = 48
num_classes = 5
epochs = 50
x_train=[]
y_train=[]
x_test=[]
y_test=[]
#files=os.listdir('jpg')
files_test=os.listdir('jpg_test')
'''

for i in files:
	if i=='.DS_Store':continue
	x_train.append(cv2.resize(cv2.imread(str('jpg/'+i)),(batch_size,batch_size)))
	y_train.append(int(i.split('-')[1].split('.')[0])-1)

'''
for i in files_test:
	if i=='.DS_Store':continue
	x_test.append(cv2.resize(cv2.imread(str('jpg_test/'+i)),(batch_size,batch_size)))
	y_test.append(int(i.split('-')[1].split('.')[0])-1)
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(x_test[idx])
plt.show()


'''

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
# 幫每個類別取名子
classes = ['1','2','3','4','5']

original_y_train = y_train.copy()
original_y_test = y_test.copy()
# 改成onehot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# 建立模型
model = Sequential()
model.add(Conv2D(48, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# 優化器
opt = keras.optimizers.rmsprop(lr=0.00001, decay=0.0000005)

# 編譯模型
model.compile(loss='categorical_crossentropy', 
              optimizer=opt, 
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test,y_test),
          shuffle=True)

model.save('ML.h5')
print(model.summary())



scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# 隨機取20個出來看看
idx = np.arange(26)

images, labels = x_test[idx], original_y_test[idx].reshape(26,)
preds = np.argmax(model.predict(x_test[idx]), axis=1)

# 畫出來
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(26):
    ax = fig.add_subplot(2, 26/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]), color=("green" if preds[idx]==labels[idx] else "red"))
plt.show()

'''
