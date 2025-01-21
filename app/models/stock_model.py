import os
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras import layers
from keras.regularizers import *
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg19 import VGG19
from keras import regularizers
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import  roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Normalization
from tensorflow.keras.applications import ResNet152V2, ResNet50V2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time
from tensorflow.keras.datasets import fashion_mnist

import tensorflow_datasets as tfds

import yfinance as yf

def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs=500):
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=50, restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
    history = model.fit(train_set, validation_data=valid_set, epochs=epochs,
                        callbacks=[early_stopping_cb])
    valid_loss, valid_mae = model.evaluate(valid_set)
    return valid_mae * 1e6

class StockDataPreprocessor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fetch_and_prepare_data(self, symbol, period='2y'):
        # Fetch data
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        # Prepare features (using Close price)
        data = df['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        return X, y, scaled_data
    
    def create_sequences(self, data):
        X = []
        y = []
        
        for i in range(self.sequence_length, len(data)):
            # Create sequence of previous 60 days
            X.append(data[i-self.sequence_length:i, 0])
            # Target is the next day's price
            y.append(data[i, 0])
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
def preprocess_data(symbol='AAPL'):
    preprocessor = StockDataPreprocessor()
    X, y, scaled_data = preprocessor.fetch_and_prepare_data(symbol)
    return X, y, scaled_data


    
