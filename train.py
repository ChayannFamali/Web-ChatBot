import random
import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from keras import callbacks 
from keras import regularizers

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open("/Web_chat_bot/intents.json").read()
intents = json.loads(data_file)

for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])
            
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)


pickle.dump(words, open("/Web_chat_bot/words.pkl", "wb"))
pickle.dump(classes, open("/Web_chat_bot/classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=1)
#weight_initializer = tf.keras.initializers.GlorotUniform()
#weight_initializer = tf.keras.initializers.HeUniform()
#weight_initializer = tf.keras.initializers.HeNormal()

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation="LeakyReLU",kernel_initializer=weight_initializer,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation="LeakyReLU",kernel_initializer=weight_initializer,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation="LeakyReLU",kernel_initializer=weight_initializer,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

optim = tf.keras.optimizers.Adam(
    learning_rate=1e-3)

model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 5, restore_best_weights = True)
learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, 
                                            factor=0.5, min_lr=0.00001)
callbacks =[earlystopping,learning_rate_reduction]

hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save("/Web_chat_bot/chatbot_model.h5", hist)
