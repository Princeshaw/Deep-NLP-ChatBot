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


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"cant't","can not",text)
    text = re.sub(r"[-(){}\"@;:<>+=~|.?,]","",text)
    return text

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))    
    
word2count = {}

for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
 
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1            
            
        
threshold = 20
questionswords2count = {}
word_count = 0
for word,count in word2count.items():  
    if count >= threshold:
        questionswords2count[word] = word_count
        word_count+=1

answerswords2count = {}
word_count = 0
for word,count in word2count.items():  
    if count >= threshold:
        answerswords2count[word] = word_count
        word_count+=1        
   
# addidng last tokens
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>'] 
for token in tokens:
    questionswords2count[token]=len(questionswords2count)+1
for token in tokens:
    answerswords2count[token]=len(answerswords2count)+1    
    
    
# create a inverse dictionary of ther answersWords2counts
answersint2words = {w_i:w for w,w_i in answerswords2count.items()}    

# adding end of the string token in the answers
for i in range(len(clean_answers)):
    clean_answers[i]+=" <EOS>"
    
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2count:
            ints.append(questionswords2count['<OUT>'])
        else:
            ints.append(questionswords2count[word])
    questions_into_int.append(ints) 


answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2count:
            ints.append(answerswords2count['<OUT>'])
        else:
            ints.append(questionswords2count[word])
    answers_into_int.append(ints)
    

sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1,25+1):
    for i in enumerate(questions_into_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])



#creating placeholderss for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32,[None,None],name='input')
    targets = tf.placeholder(tf.int32,[None,None],name='target')
    lr = tf.placeholder(tf.float32,name='learning_rate')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    return inputs,targets,lr,keep_prob

# preprocessing the targets
def preprocess_targets(targets,word2count,batch_size):
    left_side = tf.fill([batch_size,1],word2count['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    preprocessed_targets = tf.concat([left_side,right_side])
    return preprocessed_targets




        





    
    
    
    
            