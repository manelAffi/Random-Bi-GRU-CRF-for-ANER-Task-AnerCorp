from evaluationMetrics import *
import tensorflow as tf
from transformers import TFBertModel
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import keras
from keras import applications
from tensorflow.keras.layers import Dense, Input, GRU, Embedding, Dropout, Activation, Masking,LSTM, Conv1D,concatenate,Concatenate, SpatialDropout1D, GlobalMaxPooling1D,MaxPooling1D, Layer
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, TimeDistributed,Flatten
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.models import Model
from keras_contrib.layers import CRF
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
import numpy as np

max_len = 150
max_len_char = 10


#model
def create_model():
    input = Input(shape=(140,))
    word_embedding_size = 100
    output = Embedding(input_dim=29252, output_dim=word_embedding_size, input_length=140)(input)
    output = Bidirectional(GRU(units=100, return_sequences=True,recurrent_dropout=0.2, dropout=0.2))(output)
    crf = CRF(9,learn_mode="marginal")
    output = crf(output)
    #loss = losses.crf_loss
    
    model = tf.keras.models.Model(input, outputs= output)
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4),loss="categorical_crossentropy", metrics= ['accuracy',recall_m,precision_m,f1_m])
    return model