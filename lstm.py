import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense,CuDNNLSTM
from keras.datasets import mnist
from keras import optimizers
def bn_prelu(input):
    norm = BatchNormalization()(input)
    return PReLU()(norm)



class lstm:
    @staticmethod
    def lstm_model(input):
        model=Sequential()

        model.add(LSTM(128,input_shape=input,activation='relu',return_sequences=True))
        model.add(Dropout(0.3))

        model.add(LSTM(128,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32,activation='relu'))
        model.add(Dropout(0.2))



        model.add(Dense(10,activation='softmax'))
        return model


def main():

    batch_size = 128
    num_classes = 10
    epochs = 4

# input image dimensions
    img_rows, img_cols = 28, 28

# the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000,28,28)/255.0
    x_test = x_test.reshape(10000,28,28)/255.0

# convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = lstm.lstm_model((28,28))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd,metrics=['accuracy'])
    model.summary()



    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
