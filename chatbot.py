import nltk
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Nadam
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
lemmatizer = WordNetLemmatizer()
import json
import pickle
nltk.download('punkt') # This tokenizer divides a text into a list of sentences, The NLTK data package includes a pre-trained Punkt tokenizer for English.
nltk.download('wordnet')

import numpy as np

import random

#1 first process is applying NLP to our data.
words_list=[]
classes_list = []
documents_list = []
unwanted_words = ['?', '!','_','.',',','&']
full_data_file = open('job_intents.json', encoding='utf-8').read()
intents = json.loads(full_data_file) # converts into python Dictionary.

# looping through  data file.
for intent in intents['intents']:
    for pattern in intent['patterns']: # looping through patterns
        word = nltk.word_tokenize(pattern) # spliting words using nltk
        words_list.extend(word) # adding into a list
        documents_list.append((word, intent['tag'])) #  addd  pattern words   associated with tags.

        if intent['tag'] not in classes_list:
            classes_list.append(intent['tag']) # add intents to classes_list for later processing
print("Documents", documents_list)
print("words", words_list)
print("classes",classes_list)
words_list= [lemmatizer.lemmatize(w.lower()) for w in words_list if w not in unwanted_words] # lemmatize words
words_list = sorted(list(set(words_list))) #Sorted remove all duplicates. And  convert into list.
print("ist_lem_words", words_list)
print("lemmatizer words", words_list)
classes_list = sorted(list(set(classes_list)))

print (len(documents_list), "documents")

print (len(classes_list), "classes", classes_list)

print (len(words_list), "unique lemmatized words", words_list)


pickle.dump(words_list,open('words.pkl','wb'))
pickle.dump(classes_list,open('classes.pkl','wb'))

# initializing training data
training = [] # creating list for training data
output_empty = [0] * len(classes_list) # initializing  a list with all values of 0
print("output_empty",output_empty)

for d in documents_list: # enumerate
    word_bag = []
    pattern_words = d[0]
    print("pattern_words[0]",pattern_words)
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    print("pattern_words",pattern_words)
    # adding 1 if words  of word_list matches current words of pattern.
    for w in words_list:
        word_bag.append(1) if w in pattern_words else word_bag.append(0)
    output_row = list(output_empty) # creating list with zeros
    # print("output_row",output_row)
    output_row[classes_list.index(d[1])] = 1 #

    training.append([word_bag, output_row])
    # print("training",training)
# suffle  training data.
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_data_X = list(training[:,0])
train_data_Y = list(training[:,1])
print("Training data created")



# equal to number of intents to predict output intent with softmax
#
model =Sequential()
model.add(Dense(128, input_shape=(len(train_data_X[0]),), activation='relu')) # input shape is equall to length of training data.
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(len(train_data_Y[0]), activation='softmax')) # Softmax helps us to find better resposne by using Probability technique

# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['acc'])
#fitting and saving the model
history = model.fit(np.array(train_data_X), np.array(train_data_Y), epochs=200, batch_size=5, verbose=1)
# print("history of model", history.history)
results = model.evaluate(train_data_X, train_data_Y, batch_size=200)
print("test loss, test acc:", results)
model.save('chatbot_model', history)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.title("Model")
    plt.show()
plot_graphs(history,'accuracy')
plot_graphs(history,'loss')

print("model created")