from keras import Sequential
from keras.activations import relu
from keras.datasets import fashion_mnist
import tensorflow as tf
from keras.layers import MaxPooling2D, Flatten, Dense, Conv2D
from keras.optimizers import SGD
from keras.activations import sigmoid
from keras.layers import Dropout
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras import Sequential
from keras.activations import relu
from keras.callbacks import LearningRateScheduler
from keras.datasets import fashion_mnist
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import MaxPooling2D, Flatten, Dense, Conv2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical

#method to load the dataset
def load_dataset():

    (trainX, trainY), (testX,testY) = fashion_mnist.load_data()

    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, testX, trainY, testY

#method for preprocessing, such as normalizing and converting to the right type
def pre_process(train_data, test_data):
    train_norm, test_norm, trainY, testY = load_dataset()

    train_norm = train_data.astype('float32')
    test_norm = test_data.astype('float32')

    train_norm = train_norm/255.0
    test_norm = test_norm/255.0

    return train_norm, test_norm

#method to save the results of a model
def saveEpochResult(filename, history):
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save to json:
    with open(filename, mode='w') as f:
        hist_df.to_json(f)

#choice task reduce the learning rate
def reduceLearningRate(epoch):

    # reduce the learning rate every 15 epochs

    lr = 0.01

    for i in range(0,epoch):
        if(i % 15==0):
            lr *=0.1
    return  lr

#choice task data augmentation: for randomeraser see method get_random_eraser
def augementData(model,trainX,testX,trainY,testY):
    #rotation = 20 degrees, horizontalflip and randomeraser
    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=False, shear_range=0.,  preprocessing_function=get_random_eraser(v_l=0, v_h=1))
    datagen.fit(trainX)
    history = model.fit_generator(datagen.flow(trainX, trainY, batch_size=32),
                        validation_data=(testX, testY),
                        epochs=50, verbose=1, workers=4)
    hist_json_file = 'model_history_aug.json'
    saveEpochResult(hist_json_file,history)
    model.save('augumentedData_weights.hd5')

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

#method for the use of the reduced learning rate
def useReducuseLearningRate(model,trainX,testX,trainY,testY):
        callback  = LearningRateScheduler(reduceLearningRate)
        history =model.fit(trainX, trainY, epochs=50, batch_size=32, validation_data=(testX, testY), verbose=0, callbacks = [callback])
        hist_json_file = 'model_history_lr.json'
        saveEpochResult(hist_json_file,history)
        model.save('reducedLearningRate_weights.hd5')

#model 1
def model_1(aug = False,callbk = False):

    trainX, testX, trainY, testY =load_dataset()
    trainX, testX = pre_process(trainX, testX)
    input = Input(shape=(28,28,1))

    #layer1
    input_layer = Conv2D(512, (5,5), activation = 'tanh', kernel_initializer='he_uniform')(input)
    #layer2
    layer2 = MaxPooling2D(pool_size =(2, 2), strides =(2, 2))(input_layer)

    #layer 3
    layer3 = Conv2D(128, (3,3), activation="tanh", kernel_initializer='he_uniform')(layer2)
    #layer4
    layer4 = MaxPooling2D((2, 2))(layer3)

    #layer 5
    layer5 = Conv2D(64, (3,3), activation="tanh", kernel_initializer='he_uniform')(layer4)
    layer5 = Flatten()(layer5)

    #layer 6
    layer6 = Dense(100, activation='tanh', kernel_initializer='he_uniform')(layer5)
    #layer 7
    output_layer = Dense(10, activation='softmax',kernel_initializer='he_uniform')(layer6)

    model = Model(inputs=input, outputs= output_layer)
    
    opt = SGD(lr= 0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #using the reduced learning rate
    if callbk is True:
      useReducuseLearningRate(model,trainX,testX,trainY,testY)
    #using data augmentation
    elif aug is True:
      augementData(model,trainX,testX,trainY,testY)
    else:
        history =model.fit(trainX, trainY, epochs=50, batch_size=32, validation_data=(testX, testY), verbose=0)
        hist_json_file = 'model1_history.json'
        saveEpochResult(hist_json_file,history)
        model.save('model_1_Weight.hd5')
        return model


#model 2
def model_2(aug = False,callbk = False):
    
    trainX, testX, trainY, testY =load_dataset()
    trainX, testX = pre_process(trainX, testX)
    input = Input(shape=(28,28,1))
    #layer1
    input_layer = Conv2D(512, (7,7), activation = "sigmoid", kernel_initializer='he_uniform')(input)
    input_layer = Dropout(0.2)(input_layer)
    #layer 2
    layer2 = Conv2D(128, (5,5), activation="sigmoid", kernel_initializer='he_uniform')(input_layer)
    layer2 = Dropout(0.2)(layer2)
    #layer 3
    layer3 = Conv2D(64, (3,3), activation="sigmoid", kernel_initializer='he_uniform')(layer2)
    layer3 = Dropout(0.2)(layer3)
    #layer 4
    layer4 = Conv2D(64, (3,3), activation="sigmoid", kernel_initializer='he_uniform')(layer3)
    layer4 = Dropout(0.2)(layer4)
    layer4 = Flatten()(layer4)
    #layer 5
    layer5 = Dense(100, activation="sigmoid", kernel_initializer='he_uniform')(layer4)
    #layer 6
    layer6 = Dense(100, activation="sigmoid", kernel_initializer='he_uniform')(layer5)
    #layer 7
    output_layer = Dense(10, activation='softmax',kernel_initializer='he_uniform')(layer6)
    model = Model(inputs=input, outputs= output_layer)
    opt = SGD(lr= 0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    if callbk is True:
      useReducuseLearningRate(model,trainX,testX,trainY,testY)
    elif aug is True:
      augementData(model,trainX,testX,trainY,testY)
    else:
        history =model.fit(trainX, trainY, epochs=50, batch_size=32, validation_data=(testX, testY), verbose=0)
        hist_json_file = 'model2_history.json'
        saveEpochResult(hist_json_file,history)
        model.save('model_2_Weight.hd5')

#model 3
def model_3(aug = False,callbk = False):
    
    trainX, testX, trainY, testY =load_dataset()
    trainX, testX = pre_process(trainX, testX)
    input = Input(shape=(28,28,1))
    #layer1
    input_layer = Conv2D(512, (3,3), activation = 'relu', kernel_initializer='he_uniform')(input)

    #layer2
    layer2 = MaxPooling2D(pool_size =(2, 2), strides =(2, 2))(input_layer)
    layer2 = Dropout(0.2)(layer2)
    #layer 3
    layer3 = Conv2D(128, (3,3), activation="relu", kernel_initializer='he_uniform')(layer2)
    #layer4
    layer4 = MaxPooling2D((2, 2))(layer3)
    layer4 = Dropout(0.2)(layer4)
    #layer 5
    layer5 = Conv2D(64, (3,3), activation="relu", kernel_initializer='he_uniform')(layer4)
    layer5 = Flatten()(layer5)
    #layer 6
    layer6 = Dense(100, activation='relu', kernel_initializer='he_uniform')(layer5)
    #layer 7
    output_layer = Dense(10, activation='softmax',kernel_initializer='he_uniform')(layer6)
    model = Model(inputs=input, outputs= output_layer)

    opt = SGD(lr= 0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    if callbk is True:
      useReducuseLearningRate(model,trainX,testX,trainY,testY)
    elif aug is True:
      augementData(model,trainX,testX,trainY,testY)
    else:
        history =model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY), verbose=0)
        hist_json_file = 'model3_history.json'
        saveEpochResult(hist_json_file,history)
        model.save('model_3_Weight.hd5')
        prediction = model.predict(testX)
        #confusion matrix
        confusionMatrix(testY, prediction)

#choice task confusionMatrix
def confusionMatrix(testY, prediction):
    matrix = confusion_matrix(testY.argmax(axis=1), prediction.argmax(axis=1))
    print (matrix)


#choice task extra dataset
def extraTestFashionMnsistData():

    style_file = pd.read_csv("C:/Users/admin/Downloads/fashion-product-images-small/styles.csv",  error_bad_lines=False)

    test_set =list()
    test_label = list()
    dim = (28,28)
    #class 1:t_shirt nd tops
    tshirts_tops = style_file.query('articleType == "Tshirts" or articleType == "Tops" or articleType == "Innerwear Vests"')['id'].tolist()
    tshirts_tops_label = list()
    for i in range(len(tshirts_tops)):
        tshirts_tops_label.append(0)
    #read images as greyscale and resize them
    image_list_tshirtsTops = []
    for img in tshirts_tops :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_list_tshirtsTops.append(img_)
    #class 2: trousers
    trousers = style_file.query('articleType == "Trousers" ')['id'].tolist()
    trousers_label = list()
    for i in range(len(trousers)):
        trousers_label.append(1)
    #read images as greyscale and resize them
    image_trousers= []
    for img in trousers :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_trousers.append(img_)
    #class 3:pullovers
    pullovers = style_file.query('articleType == "Sweaters" ')['id'].tolist()
    pullovers_label = list()
    for i in range(len(pullovers)):
        pullovers_label.append(2)
    #read images as greyscale and resize them
    image_pullovers= []
    for img in pullovers :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_pullovers.append(img_)
    #class 4:Dress
    dress = style_file.query('articleType == "Dresses" or articleType == "Nightdress"')['id'].tolist()
    dress_label = list()
    for i in range(len(dress)):
        dress_label.append(3)
    #read images as greyscale and resize them
    image_dress= []
    for img in dress :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_dress.append(img_)
    #class 5:Coat
    coat = style_file.query('articleType == "Waistcoat" or articleType == "Jackets"')['id'].tolist()
    coat_label = list()
    for i in range(len(coat)):
        coat_label.append(4)
    #read images as greyscale and resize them
    image_coat= []
    for img in coat :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_coat.append(img_)
    #class 6:Sandals
    sandals = style_file.query('articleType == "Sandals"')['id'].tolist()
    sandals_label = list()
    for i in range(len(sandals)):
        sandals_label.append(5)
    #read images as greyscale and resize them
    image_sandals= []
    for img in sandals :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_sandals.append(img_)
    #class 7:Shirts
    shirts = style_file.query('articleType == "Shirts"')['id'].tolist()
    shirts_label = list()
    for i in range(len(shirts)):
        shirts_label.append(6)
    #read images as greyscale and resize them
    image_shirts= []
    for img in shirts :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_shirts.append(img_)
     #class 8:Sneakers
    sneakers = style_file.query('articleType == "Sports Shoes" or articleType == "Casual Shoes"')['id'].tolist()
    sneakers_label = list()
    for i in range(len(sneakers)):
        sneakers_label.append(7)
    #read images as greyscale and resize them
    image_sneakers= []
    for img in sneakers :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_sneakers.append(img_)
    #class 9:Bags
    bags = style_file.query('subCategory == "Bags"')['id'].tolist()
    bags_label = list()
    for i in range(len(bags)):
        bags_label.append(8)
    #read images as greyscale and resize them
    image_bags= []
    for img in bags :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_bags.append(img_)
    #class 10: ankle boot
    boots = style_file[style_file["productDisplayName"].str.contains("Ankle Boots",na=False)]['id'].tolist()
    boots_label = list()
    for i in range(len(boots)):
        boots_label.append(9)
    #read images as greyscale and resize them
    image_boots= []
    for img in boots :
        filename ='fashion-product-images-small/images/'+ str(img)+".jpg"
        im=cv2.imread(filename,0)
        try:
            img_ = cv2.resize(im, dim)
        except Exception as e:
            print(str(e))
        image_boots.append(img_)
    #combine labels to one flat list
    test_label.append(tshirts_tops_label)
    test_label.append(trousers_label)
    test_label.append(pullovers_label)
    test_label.append(dress_label)
    test_label.append(coat_label)
    test_label.append(sandals_label)
    test_label.append(shirts_label)
    test_label.append(sneakers_label)
    test_label.append(bags_label)
    test_label.append(boots_label)
    #flat_label = list(np.array(test_label))
    flat_label = [item for sublist in test_label for item in sublist]
    flat_label = np.asarray(flat_label)
    flat_label = to_categorical(flat_label)
    #combine the images to one single list and convert to nd array
    test_set.append(image_list_tshirtsTops)
    test_set.append(image_trousers)
    test_set.append(image_pullovers)
    test_set.append(image_dress)
    test_set.append(image_coat)
    test_set.append(image_sandals)
    test_set.append(image_shirts)
    test_set.append(image_sneakers)
    test_set.append(image_bags)
    test_set.append(image_boots)
    #reshape and normalize data
    flat_list = [item for sublist in test_set for item in sublist]
    flat_list = np.asarray(flat_list)
    flat_list= flat_list.reshape((flat_list.shape[0], 28, 28, 1))
    flat_list = flat_list.astype('float32')
    flat_list = flat_list/255.0
    return flat_list,flat_label

def main():
    #run model1 without any choice task
    #model_1()

    #run model2 without any choice task
    #model_2()

    #run model3 without any choice task


    #run model 3 with augumented_data
    #model_3(True, False)

    #run model 3 with reducing learning rate
    #model_3(False, True)

    #run on new imagedataset(kaggle fashion dataset)
    #model = model_1()
    #f_list, f_label = extraTestFashionMnsistData()
    #test_loss, test_acc = model.evaluate(f_list, f_label, verbose=2)
    #print(test_loss, test_acc)



if __name__ == '__main__':
    main()
