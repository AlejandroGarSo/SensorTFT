import numpy as np
import pandas as pd
import keras
import utils
from keras.layers import Input,Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def train(model, epochs, data, labels, iter):
    model.compile(optimizer="sgd", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=15, restore_best_weights=True)

    for i in range(iter):
        print("Test " + str(i+1))
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
        history = model.fit(train_data, train_labels, epochs=epochs, validation_data=[test_data, test_labels],
                            callbacks=[es])

def getConfMatrix(model, data, labels):
    pred = np.argmax(model.predict(data), axis=-1)
    confusion_matrix(labels, pred)
    print(confusion_matrix)

def loadModel(modelFile):
    return keras.models.load_model(modelFile)

def saveModel(model, modelFile):
    model.save(modelFile)

def predict(model, data):
    #size = model.input.input_shape[1]
    size = 854
    d = utils.flattenData(data, size)
    return np.argmax(model.predict(np.expand_dims(d, axis=0)), axis=-1)

def createModel(input):
    inputs = Input(shape=(input * 3))
    hidden = Dense(units=10)(inputs)
    output = Dense(units=10, activation='softmax')(hidden)

    model = keras.Model(inputs=inputs, outputs=output)
    return model