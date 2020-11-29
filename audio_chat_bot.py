# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:26:08 2020

@author: pitonhik
"""

import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
# Словарь
def clean_str(r):
    r = r.lower()
    r = [c for c in r if c in alphabet]
    return ''.join(r)
alphabet = ' 1234567890-йцукенгшщзхъфывапролджэячсмитьбюёqwertyuiopasdfghjklzxcvbnm'
with open('en2.txt', encoding='utf-8') as f:
    content = f.read()
blocks = content.split('\n')
dataset = []
for block in blocks:
    replicas = block.split('\\')[:2]
    if len(replicas) == 2:
        pair = [clean_str(replicas[0]), clean_str(replicas[1])]
        if pair[0] and pair[1]:
            dataset.append(pair)
X_text = []
y = []
for question, answer in dataset[:10000]:
    X_text.append(question)
    y += [answer]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)
clf = LogisticRegression()
clf.fit(X, y)
def get_generative_replica(text):
    text_vector = vectorizer.transform([text])
    text_vector = text_vector.toarray()[0]
    question = clf.predict([text_vector])
    question = question[0]
    return question
# Голосовой ассистент
def listen():
    voice_recognizer = sr.Recognizer()
    #voice_recognizer.dynamic_energy_threshold = False
    #voice_recognizer.energy_threshold = 1000
    #voice_recognizer.pause_threshold = 0.5
    with sr.Microphone() as source:
        print("Говорите:")
        audio = voice_recognizer.listen(source, timeout = None, phrase_time_limit = 2)
    try:
        vt = voice_recognizer.recognize_google(audio)
        
        print(f"Вы сказали: {vt}")
        return vt
    except sr.UnknownValueError:
        return "Ошибка распознания"
    except sr.RequestError:
        return "Ошибка соединения"
def say(text):
    voice = gTTS(text, lang="En")
    unique_file = "audio_" + str(random.randint(0, 10000)) + ".mp3"
    voice.save(unique_file)
    playsound.playsound(unique_file)
    os.remove(unique_file)
    print(f"Бот:  {text}")
def handle_command(command):
    command = command.lower()
    reply = get_generative_replica(command)
    say(reply)
def stop():
    say("Goodbye!")
def start():
    say('hello my friend')
    print(f"Запуск бота...")
    while True:
        command = listen()
        handle_command(command)
try:
    start()
except KeyboardInterrupt:
    stop()