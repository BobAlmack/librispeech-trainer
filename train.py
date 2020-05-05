import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

standardize_mean, standardize_std = np.load("standardization.npy")

def load_label(label):
    global standardize_mean
    global standardize_std
    dxlen = np.load(label + "xlen.npy").astype("float32").reshape((-1, 1))
    dx = np.load(label + "x.npy").astype("float32")
    dx = (dx - standardize_mean) / standardize_std
    dylen = np.load(label + "ylen.npy").astype("float32").reshape((-1, 1))
    dy = np.load(label + "y.npy").astype("float32")
    blanklabel = np.zeros((dy.shape[0], dy.shape[1], 1))
    dy = np.concatenate([dy, blanklabel], axis=-1)
    p = np.random.permutation(len(dxlen))
    dxlen = dxlen[p]
    dx = dx[p]
    dylen = dylen[p]
    dy = dy[p]
    return dxlen, dx, dylen, dy

train_xlen, train_x, train_ylen, train_y = load_label("train")
test_xlen, test_x, test_ylen, test_y = load_label("test")

print(train_xlen.shape)
print(train_x.shape)
print(train_ylen.shape)
print(train_y.shape)
print(test_xlen.shape)
print(test_x.shape)
print(test_ylen.shape)
print(test_y.shape)

def ctc_lambda_loss(args):
    y_pred, labels, wavlen, lablen = args
    labels = K.argmax(labels)
    return K.ctc_batch_cost(labels, y_pred, wavlen, lablen)

def create_model():
    input_wavlen = Input((1,))
    input_wav = Input((3150, 26))
    input_lablen = Input((1,))
    input_lab = Input((400, 42))
    x = Bidirectional(LSTM(100, return_sequences=True))(input_wav)
    x = LayerNormalization()(x)
    x = TimeDistributed(Dense(42, activation="softmax"))(x)
    xl = Lambda(ctc_lambda_loss, output_shape=(1,))([x, input_lab, input_wavlen, input_lablen])
    m = Model([input_wavlen, input_wav, input_lablen, input_lab], xl)
    m.summary()
    m.compile(loss={'lambda': lambda y_true, y_pred: y_pred}, optimizer=tf.keras.optimizers.Adam(lr=1e-4))
    return m

if __name__ == "__main__":
    strat = tf.distribute.MirroredStrategy()
    with strat.scope():
        m = create_model()
    m.fit([train_xlen, train_x, train_ylen, train_y], np.zeros((len(train_y), 1)), validation_data=([test_xlen, test_x, test_ylen, test_y], np.zeros((len(test_y), 1))), batch_size=512, epochs=120, verbose=1)
    m.save("model.h5")
