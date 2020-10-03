#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import time

# importing dataset
lines = open("./dataset/movie_lines.txt",encoding="utf-8",errors="ignore").read().split('\n')
conversations = open('./dataset/movie_conversations.txt',encoding="utf-8",errors="ignore").read().split('\n')

id2lines = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line)==5:
        id2lines[_line[0]]= _line[4]


conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2lines[conversation[i]])
        answers.append(id2lines[conversation[i+1]])
        