from DataReader import *
from model import *
from  test import *
from collections import Counter
from future.utils import iteritems
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pydot
#read data
ANERCorp_path = "/home/mrim/affim/Random-BGC-For-Arabic/Data/ANERcorp-CamelLabSplits/"
data_train = read_ANERcorp(ANERCorp_path+'ANERCorp_CamelLab_train.txt')
data_test = read_ANERcorp(ANERCorp_path+'ANERCorp_CamelLab_test.txt')
#read label
print(Counter([ label for sentence in data_test for label in sentence[1]]))
print(Counter([ label for sentence in data_train for label in sentence[1]]))
label_list = list(Counter([ label for sentence in data_test for label in sentence[1]]).keys())
word_list = list(Counter([ word for sentence in data_train for word in sentence[0]]).keys())
word_list_test = list(Counter([ word for sentence in data_test for word in sentence[0]]).keys())
#print(data_train[0])
#print(label_list)
tags=[]
words=[]
#read token and label list from data train and data test
for tag in data_train:
    tags.append(tag[1])

for word in data_train:
    words.append(word[0])
tags_test=[]
words_test=[]
for tag in data_test:
    tags_test.append(tag[1])
 
for word in data_test:
    words_test.append(word[0])

from keras.preprocessing.sequence import pad_sequences

tag=[]
word2idx = {w: i for i, w in enumerate(word_list)}  
word2idxtest = {w: i for i, w in enumerate(word_list_test)}  
X = [[word2idx[w] for w in word] for word in words]
words= pad_sequences(X, dtype=object, maxlen=140, value=len(word_list) - 1,padding='post')

tag2idx = {t: i for i, t in enumerate(label_list)}
idx2tag = {v: k for k, v in iteritems(tag2idx)}
y_idx = [[tag2idx[w] for w in tag] for tag in tags]
tag= pad_sequences(y_idx, dtype=object, maxlen=140, value=tag2idx["O"],padding='post')
y = [to_categorical(i, num_classes=9) for i in tag]
tag=[]  
X_test = [[word2idxtest[w] for w in word] for word in words_test]
words_test= pad_sequences(X_test, dtype=object, maxlen=140, value=len(word_list) - 1,padding='post')
#print(np.array(words_test).shape)
#print(tags[0])
y_idx_test = [[tag2idx[w] for w in tag] for tag in tags_test]
tag= pad_sequences(y_idx_test, dtype=object, maxlen=140, value=tag2idx["O"],padding='post')
y_test = [to_categorical(i, num_classes=9) for i in tag]


#print(x[0])
#train the model

model = create_model()
model.summary()
"""from keras.utils.vis_utils import plot_model
tf.keras.utils.plot_model(
model, to_file='/home/mrim/affim/BCE-EADC-For-Arabic/model.png', show_shapes=False, show_dtype=False,
show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)
from plot_model import plot_model
plot_model(model, to_file='/home/mrim/affim/BCE-EADC-For-Arabic/model.png')"""

es = EarlyStopping(monitor='val_accuracy', mode='max',  verbose=1, patience=10)
"""history = model.fit(np.asarray(words).astype('float32'),
    np.asarray(y).astype('float32'), 
    validation_split = 0.2,
    epochs = 50,
    batch_size = 64,
    callbacks=[es]
)
save_model(model)"""
test_model(np.asarray(words_test).astype('float32'), np.asarray(y_test).astype('float32'))