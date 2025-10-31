import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Embedding, Dropout, Activation, Dense, BatchNormalization, Bidirectional, GRU, Layer, Conv1D, \
    Concatenate, AvgPool1D, Flatten
from keras.metrics import AUC
from keras.optimizers import Adam

one_hot = np.array([[0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])


class SelfAttention(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = tf.compat.v1.keras.initializers.RandomNormal(seed=42)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weight = [self.W, self.b, self.u]
        super(SelfAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def dense_block(x, num_layers, growth_rate, dropout_rate=0.1, kernel_size=3):
    features = [x]
    for _ in range(num_layers):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(growth_rate, kernel_size, padding="same")(x)
        x = Dropout(dropout_rate)(x)
        features.append(x)
        x = Concatenate(axis=-1)(features)
    return x


def transition_layer(x, compression=0.5, pool_size=2):
    x = BatchNormalization()(x)
    x = Conv1D(int(x.shape[-1] * compression), 1, padding="same")(x)  # 通道压缩
    x = AvgPool1D(pool_size, strides=pool_size, padding="same")(x)  # 降采样
    return x


def CNN_GRU_ATT_model():
    sequence = Input(shape=(174,))
    x = Embedding(5, 4, weights=[one_hot], trainable=False)(sequence)

    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = AvgPool1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = AvgPool1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    gru = Bidirectional(GRU(16, return_sequences=True))(x)
    att = SelfAttention(16)(gru)
    bn = BatchNormalization()(att)
    dt = Dropout(0.2)(bn)
    dt = Dense(64)(dt)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    md = Model([sequence], preds)
    md.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    return md


def CNN_GRU_model():
    sequence = Input(shape=(174,))
    x = Embedding(5, 4, weights=[one_hot], trainable=False)(sequence)

    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = AvgPool1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = AvgPool1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    gru = Bidirectional(GRU(16, return_sequences=True))(x)
    bn = BatchNormalization()(gru)
    flt = Flatten()(bn)
    dt = Dropout(0.2)(flt)
    dt = Dense(64)(dt)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    md = Model([sequence], preds)
    md.compile(
        optimizer=Adam(learning_rate=0.0002),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    return md


def CNN_model():
    sequence = Input(shape=(174,))
    x = Embedding(5, 4, weights=[one_hot], trainable=False)(sequence)

    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
    x = AvgPool1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = AvgPool1D(pool_size=2)(x)
    bn = BatchNormalization()(x)

    flt = Flatten()(bn)
    dt = Dropout(0.2)(flt)
    dt = Dense(64)(dt)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    md = Model([sequence], preds)
    md.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    return md


def StackIRESFinding():
    sequence = Input(shape=(174,))
    x = Embedding(5, 4, weights=[one_hot],
                  trainable=False)(sequence)

    for _ in range(3):
        x = dense_block(x, 3, 48, 0.27, 3)
        x = transition_layer(x, 0.27, 3)
    x = dense_block(x, 3, 48, 0.27, 3)

    gru = Bidirectional(GRU(18, return_sequences=True))(x)
    att = SelfAttention(attention_dim=12)(gru)

    dt = Dense(256)(att)
    preds = Dense(1, activation='sigmoid')(dt)

    model = Model([sequence], preds)
    model.compile(
        optimizer=Adam(learning_rate=0.0011),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    return model
