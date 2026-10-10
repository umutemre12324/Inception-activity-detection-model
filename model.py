from tensorflow.keras import Model
from tensorflow.keras.layers import Input ,Activation, Conv1D, MaxPooling1D,GlobalAveragePooling1D, Dense, Dropout, BatchNormalization,Concatenate
import numpy as np
import joblib

def inception_module(x,filters=32):

    conv1=Conv1D(filters=filters,kernel_size=1,padding="same",activation="relu")(x)
    conv3=Conv1D(filters=filters,kernel_size=3,padding="same",activation="relu")(x)
    conv5=Conv1D(filters=filters,kernel_size=5,padding="same",activation="relu")(x)
    maxpool=MaxPooling1D(pool_size=3,strides=1,padding="same")(x)

    pool=Conv1D(filters=filters,kernel_size=1,padding="same",activation="relu")(maxpool)
    output=Concatenate(axis=-1)([conv1,conv3,conv5,pool])
    return output

def build_inception_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = inception_module(inputs, filters=32)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = inception_module(x, filters=64)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = inception_module(x, filters=128)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__": 
    
    # Load preprocessed data
    x_train_scaled = np.load("x_train_scaled.npy")
    y_train_to = np.load("y_train_to.npy")
    x_test_scaled = np.load("x_test_scaled.npy")
    y_test_to = np.load("y_test_to.npy")

    # Get input shape and number of classes
    input_shape = x_train_scaled.shape[1:]  # (timesteps, channels)
    num_classes = y_train_to.shape[1]

    # Build the model
    model = build_inception_model(input_shape, num_classes)
    model.summary()

