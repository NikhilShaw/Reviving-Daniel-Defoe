import numpy as np 
import sys
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
text = open("Robinson_crusoe.txt").read().lower()
chars_ = sorted(list(set(text)))
# trimming characters except alphanum and breakers
chars=[]
for x in chars_:
    if x =='z':
        chars.append(x)
        break
    chars.append(x)
text_mod=""
for x in range(len(text)):
    if text[x] in chars:
        text_mod+=text[x]
# creating mapping of unique characters
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
seq_length= 100
dataY = []
dataX = []
for i in range(0, len(text_mod)- seq_length, 1):
    seq_in = text_mod[i:i+seq_length]
    seq_out = text_mod[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append([char_to_int[char] for char in seq_out])
# reshaping X to be [samples, time steps, features]
x = np.reshape(dataX, (len(dataX), seq_length, 1))
#normalize
x=x/float(len(chars))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' )
# define the checkpoint
filepath="best_weight.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor= 'loss' , verbose=1, save_best_only=True,
mode= min )
callbacks_list = [checkpoint]
# fit the model
model.fit(x, y, nb_epoch=50, batch_size=64, callbacks= callbacks_list)
# load the network weights
filename = "best_weight.hdf5"
model.load_weights(filename)
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' )
# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"",''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print "\nDone."
