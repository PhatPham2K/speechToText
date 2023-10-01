import datetime
import json
import pickle
import random
import pickle
import nltk
import numpy as np
import time
import face_recognition as fr
import pyttsx3
from nltk import WordNetLemmatizer
from keras.models import load_model



wnl = WordNetLemmatizer();

file_path_json = 'static/intents.json'
file_path_word = 'static/word.pkl'
file_path_classes = 'static/classes.pkl'
file_path_model = 'static/chatbot_model_new.h5'

intents = json.load(open(file_path_json))
words = pickle.load(open(file_path_word,'rb'))
classes = pickle.load(open(file_path_classes,'rb'))
model = load_model(file_path_model)

leimatizer = WordNetLemmatizer()

def cleanup_sentense(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [leimatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = cleanup_sentense(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    p = bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]

    CONFIDENT_SCORE = 0.7
    results = [[i, r] for i, r in enumerate(res) if r > CONFIDENT_SCORE]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def getResponse(intent_list, intent_json):
    flag = True
    result = ""
    try:
        tag = intent_list[0]['intent']
        print("Tag ...", tag)
        list_of_intents = intent_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                if tag == 'currenttime':
                    t = time.localtime()
                    current_time = time.strftime("%H:%M", t)
                    hour  = time.strftime("%H")


                    h = int(hour)

                    stage = " a m "
                    if h >= 12:
                        stage = " p m "
                    result = random.choice(i['responses']) + current_time + stage
                    # print(result)
                elif i['tag'] == 'dayinmonth':
                    now = datetime.date.today() # current date and time

                    year = now.strftime("%Y")
                    month = now.strftime("%m")
                    day = now.strftime("%d")
                    dateTime = day+"/"+month+"/"+year
                    result = random.choice(i['responses']) + dateTime
                elif i['tag'] == 'goodbye':
                    result = random.choice(i['responses'])
                    flag = False
                elif i['tag'] == 'face_detection':
                    result = random.choice(i['responses'])
                    fr.faceDetection()
                else:
                    result = random.choice(i['responses'])
                break
    except IndexError:
        result = "I don't understand!"
    return result, flag

# ints = predict_class("How are you?")
# res  = getResponse(ints,intents)
#
#
# while True:
#     message = input("")
#     ints = predict_class(message)
#     res  = getResponse(ints,intents)
#     print(res)
#     Main.SpeakText(res)


