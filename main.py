import matplotlib.pyplot as plt
import cv2
import numpy as np

import keras
import os, re, time, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt

print(tf.version.VERSION)

imgSIZE = np.asarray((320, 240))

def pltshow(IMG, ra):
    r = ra * img.shape
    img = IMG.copy()
    cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (1, 0, 0), 1)
    plt.imshow(img)
    plt.show()

def MODEL():
    optimizer = keras.optimizers.Adam(learning_rate=0.0002)
    model = models.Sequential()
    model.add(layers.Conv2D(2, (3,3), activation='relu', input_shape=(imgSIZE[0], imgSIZE[1], 1)))
    model.add(layers.Dropout(0.05))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.08))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.05))
    #model.add(layers.Dense(units = 128, name = 'dense0'))
    model.add(layers.Dense(units = 4, name = 'bounding_box'))

    #v4
    #model.add(layers.Conv2D(10, (3,3), activation='relu', input_shape=(imgSIZE[0], imgSIZE[1], 1)))
    #model.add(layers.Dropout(0.02))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(12, (3, 3), activation='relu'))
    #model.add(layers.Dropout(0.05))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(6, (3, 3), activation='relu'))
    #model.add(layers.Dropout(0.03))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Flatten())
    #model.add(layers.Dense(units = 4, name = 'bounding_box'))

    #model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=(imgSIZE[0], imgSIZE[1], 1)))
    #model.add(layers.Dropout(0.02))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    #model.add(layers.Dropout(0.05))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    #model.add(layers.Dropout(0.03))
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Flatten())
    #model.add(layers.Dense(units = 4, name = 'bounding_box'))

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error', #binary_crossentropy, mean_squared_error
                  metrics=['accuracy'])

    model.summary()
    return model

def data():
    labl = "image/label.json"
    labl = json.load(open(labl))
    imgs = []
    label = []
    for p in labl: #get images
        img = np.asarray(cv2.cvtColor(cv2.imread("image/" + p), cv2.COLOR_BGR2GRAY), dtype=np.float32) / 255.0
        img = cv2.resize(img, imgSIZE)
        ROI = img[img.shape[0]//2-25:img.shape[0]//2+25, img.shape[1]//2-25:img.shape[1]//2+25]
        h = cv2.calcHist([ROI],[0],None,[256],[0,1])
        h = np.cumsum(h)
        h = h / h[-1]
        ha = (h > 0.015) * (h < 0.975) #97%
        m = max(np.argmax(ha) / 256 - 30/256, 0)
        M = min((256-np.argmax(ha[::-1])) / 256 + 2/256, 1)
        imga = (np.asarray(img) - m) / (M - m)
        imga[imga < 0] = 0
        imga[imga > 1] = 1
        if False:
            plt.subplot(221)
            plt.imshow(ROI)
            plt.subplot(222)
            plt.imshow(img)
            plt.subplot(223)

            print(m, M, 220/256)
            print( "test", (220/256 - m) / (M - m) )
            plt.plot(h)
            plt.plot(ha)
            plt.subplot(224)

            plt.imshow(imga)
            plt.show()
            if np.random.rand() > 0.7:
                exit()
        imgs.append(imga.copy())

        l = np.asarray(labl[p], dtype=np.float32)#.flatten()

        l[:,0] /= 640
        l[:,1] /= 480

        #x, y, x, y
        l[0,0], l[1,0] = min(l[:,0]), max(l[:,0])
        l[0,1], l[1,1] = min(l[:,1]), max(l[:,1])

        #to x,y,w,h
        l[1,:] = l[1,:] - l[0,:] + np.random.rand(2) * 5/256
        l[0,:] = l[0,:] - np.random.rand(2) * 5/256
        l = l.flatten()
        #pltshow(img, l)
        #exit()
        label.append(l.copy())

        imga = imga[::-1,::-1]
        #print(l)
        l[:2] = 1-l[:2] - l[-2:]
        #print(l)
        #exit()
        imgs.append(imga)
        label.append(l.copy())
    return imgs, label

imgs, label = data()
print(len(imgs), len(label))
print("image shape:", imgs[0].shape)
train_images = np.asarray(imgs[:-7])
test_images = np.asarray(imgs[-7:])
train_labels = np.asarray(label[:-7])
test_labels = np.asarray(label[-7:])
 
model = MODEL()
try:
    model.load_weights("MODEL_W.weights.h5")
    print("LOADED")
except Exception as e:
    print(e)
history = model.fit(train_images, train_labels, epochs=25, batch_size=15, validation_data=(test_images, test_labels))
model.save_weights('MODEL_W.weights.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")

fig = plt.figure()
for i in range(7):
    img = np.asarray(test_images[i])
    r = np.asarray(test_labels[i]).reshape(2,2)

    r = r * imgSIZE
    img_input = np.expand_dims(img, axis=0)
    p = np.uint16(model.predict(img_input)[0].reshape(2,2) * imgSIZE)
    print("p", p)
    print("img", img.shape)

    r = np.uint(r.flatten())
    p = np.uint(p.flatten())
    cv2.rectangle(img, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (1, 0, 0), 1)
    cv2.rectangle(img, (p[0], p[1]), (p[0]+p[2], p[1]+p[3]), (1, 0, 0), 2)
    ax = fig.add_subplot(331+i)
    ax.imshow(img)
plt.show()
