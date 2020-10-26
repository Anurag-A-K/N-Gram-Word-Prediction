import nltk as nltk
from nltk.util import ngrams
from collections import defaultdict
from collections import OrderedDict
import string
import json
import sys
import os

# Preprocessing - Remove punctuations
def removePunctuations(sen):
    temp_l = sen.split()
    i,j = 0,0
    
    for word in temp_l :
        j = 0
        for l in word :
            if l in string.punctuation:
                if l == "'":
                    if j+1<len(word) and word[j+1] == 's':
                        j += 1
                        continue
                word = word.replace(l," ")
            j += 1

        temp_l[i] = word.lower()
        i += 1   
    content = " ".join(temp_l)
    return content


# Loading Corpus
def loadCorpus(file_path, bi_dict, tri_dict, quad_dict, vocab_dict):

    w1,w2,w3 = '','',''     
    token, word_len = [], 0

    file = open(file_path,'r',encoding="utf8")
    for line in file:
        temp_l = line.split()
        i, j = 0, 0
        for word in temp_l :
            j = 0
            for l in word :
                if l in string.punctuation:
                    if l == "'":
                        if j+1<len(word) and word[j+1] == 's':
                            j += 1
                            continue
                    word = word.replace(l," ")
                j += 1
            temp_l[i] = word.lower()
            i += 1   
   
        content = " ".join(temp_l)

        token = content.split()
        word_len = word_len + len(token)  

        if not token:
            continue

        if w3!= '':
            token.insert(0,w3)

        temp0 = list(ngrams(token,2))

        if w2!= '':
            token.insert(0,w2)

        temp1 = list(ngrams(token,3))

        if w1!= '':
            token.insert(0,w1)

        for word in token:
            if word not in vocab_dict:
                vocab_dict[word] = 1
            else:
                vocab_dict[word]+= 1
                  
        temp2 = list(ngrams(token,4))

        #Bigram Frequency
        for t in temp0:
            sen = ' '.join(t)
            bi_dict[sen] += 1

        #Trigram Frequency
        for t in temp1:
            sen = ' '.join(t)
            tri_dict[sen] += 1

        #Quadgram Frequency
        for t in temp2:
            sen = ' '.join(t)
            quad_dict[sen] += 1


        n = len(token)

        #Storing words for sentence pairing
        if (n -3) >= 0:
            w1 = token[n -3]
        if (n -2) >= 0:
            w2 = token[n -2]
        if (n -1) >= 0:
            w3 = token[n -1]
    return word_len

def findQuadgramProbGT(vocab_dict, bi_dict, tri_dict, quad_dict, quad_prob_dict, nc_dict, k):    
    V, i = len(vocab_dict), 0
    for quad_sen in quad_dict:
        quad_token = quad_sen.split()
        
        #Trigram for key
        tri_sen = ' '.join(quad_token[:3])

        #Good Turing Smoothing
        quad_count = quad_dict[quad_sen]
        tri_count = tri_dict[tri_sen]
        
        if quad_dict[quad_sen] <= k  or (quad_sen not in quad_dict):
            quad_count = findGoodTuringAdjustCount( quad_dict[quad_sen], k, nc_dict)
        if tri_dict[tri_sen] <= k  or (tri_sen not in tri_dict):
            tri_count = findGoodTuringAdjustCount( tri_dict[tri_sen], k, nc_dict)
        
        prob = quad_count / tri_count
        
        #Add the Trigram to the Quadgram Probabiltity Dictionary
        if tri_sen not in quad_prob_dict:
            quad_prob_dict[tri_sen] = []
            quad_prob_dict[tri_sen].append([prob,quad_token[-1]])
        else:
            quad_prob_dict[tri_sen].append([prob,quad_token[-1]])
  
    prob = None
    quad_token = None
    tri_sen = None

def findTrigramProbGT(vocab_dict, bi_dict, tri_dict, tri_prob_dict, nc_dict, k):

    V = len(vocab_dict)
    for tri in tri_dict:
        tri_token = tri.split()

        bi_sen = ' '.join(tri_token[:2])
        
        tri_count = tri_dict[tri]
        bi_count = bi_dict[bi_sen]
        
        if tri_dict[tri] <= k or (tri not in tri_dict):
            tri_count = findGoodTuringAdjustCount( tri_dict[tri], k, nc_dict)
        if bi_dict[bi_sen] <= k or (bi_sen not in bi_dict):
            bi_count = findGoodTuringAdjustCount( bi_dict[bi_sen], k, nc_dict)
        
        prob = tri_count / bi_count
        
        if bi_sen not in tri_prob_dict:
            tri_prob_dict[bi_sen] = []
            tri_prob_dict[bi_sen].append([prob,tri_token[-1]])
        else:
            tri_prob_dict[bi_sen].append([prob,tri_token[-1]])
    
    prob = None
    tri_token = None
    bi_sen = None

def findBigramProbGT(vocab_dict, bi_dict, bi_prob_dict, nc_dict, k):
   
    V = len(vocab_dict)
    
    for bi in bi_dict:
        bi_token = bi.split()
        unigram = bi_token[0]
       
        bi_count = bi_dict[bi]
        uni_count = vocab_dict[unigram]
        
        if bi_dict[bi] <= k or (bi not in bi_dict):
            bi_count = findGoodTuringAdjustCount( bi_dict[bi], k, nc_dict)
        if vocab_dict[unigram] <= k or (unigram not in vocab_dict):
            uni_count = findGoodTuringAdjustCount( vocab_dict[unigram], k, nc_dict)
        
        prob = bi_count / uni_count

        if unigram not in bi_prob_dict:
            bi_prob_dict[unigram] = []
            bi_prob_dict[unigram].append([prob,bi_token[-1]])
        else:
            bi_prob_dict[unigram].append([prob,bi_token[-1]])
    
   
    prob = None
    bi_token = None
    unigram = None

def sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict):
    for key in bi_prob_dict:
        if len(bi_prob_dict[key])>1:
            bi_prob_dict[key] = sorted(bi_prob_dict[key],reverse = True)
    
    for key in tri_prob_dict:
        if len(tri_prob_dict[key])>1:
            tri_prob_dict[key] = sorted(tri_prob_dict[key],reverse = True)
    
    for key in quad_prob_dict:
        if len(quad_prob_dict[key])>1:
            quad_prob_dict[key] = sorted(quad_prob_dict[key],reverse = True)[:2]


def processInput(sentence):
    sen = sentence        
    sen = removePunctuations(sen)
    temp = sen.split()
    if len(temp) < 3:
        return "Invalid"
    else:
        cond = True
        temp = temp[-3:]
        sen = " ".join(temp)
        return sen

#Backoff
def computeTestScore(test_token, bi_dict, tri_dict, quad_dict, quad_prob_dict, tri_prob_dict,bi_prob_dict ):
    
    score, wrong, total = 0, 0, 0
    w = open('server-log-outputs/Good_Turing_Backoff_Score.txt','w')
    for sent in test_token:
        sen_token = sent[:3]
        sen = " ".join(sen_token)
        correct_word = sent[3]
            
        result = doPredictionBackoffGT(sen, bi_dict, tri_dict, quad_dict, bi_prob_dict, tri_prob_dict, quad_prob_dict)
        if result:
            if result[1] == correct_word:
                score+=1
            else:
                wrong += 1
        else:
            wrong += 1
        total += 1
            
    w.write('Total Word Prdictions: '+str(total) + '\n' +'Correct Prdictions: '+str(score) +'\n'+'Wrong Prdictions: '+str(wrong) + '\n'+'ACCURACY: '+str((score/total)*100)+'%' )
    return score

def chooseWords(sen, bi_prob_dict, tri_prob_dict, quad_prob_dict):
    word_choice = []
    token = sen.split()
    if token[-1] in bi_prob_dict:
        word_choice +=  bi_prob_dict[token[-1]][:1]
    if ' '.join(token[1:]) in tri_prob_dict:
        word_choice +=  tri_prob_dict[' '.join(token[1:])][:1]
    if ' '.join(token) in quad_prob_dict:
        word_choice += quad_prob_dict[' '.join(token)][:1]
    
    return word_choice

#BackOff
def computePerplexity(test_quadgrams, bi_dict, tri_dict, quad_dict, vocab_dict,token_len, k, quad_nc_dict, tri_nc_dict,bi_nc_dict, uni_nc_dict):
    
    perplexity = float(1.0)
    n = token_len
    for key in quad_dict:
        quad_token = key.split()
        quad_count = quad_dict[key]
        tri_count = tri_dict[' '.join(quad_token[0:3])]
        
        if quad_dict[key] <= k or (key not in quad_dict):
            quad_count = findGoodTuringAdjustCount( quad_dict[key], k, quad_nc_dict)
        if tri_dict[' '.join(quad_token[0:3])] <= k  or (' '.join(quad_token[0:3]) not in tri_dict):
            tri_count = findGoodTuringAdjustCount( tri_dict[' '.join(quad_token[0:3])], k, tri_nc_dict)
        prob = quad_count / tri_count
        
        if prob != 0:
            perplexity = perplexity * ( prob**(1./n))

    w = open('server-log-outputs/Good_Turing_Backoff_Score.txt','a')
    w.write('\nPerplexity:'+str(perplexity))
    return perplexity


from statistics import mean
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style

#finds the slope for the best fit line
def findBestFitSlope(x,y):
    m = (( mean(x)*mean(y) - mean(x*y) ) / ( mean(x)** 2 - mean(x**2)))
    return m
      
#finds the intercept for the best fit line
def findBestFitIntercept(x,y,m):
    c = mean(y) - m*mean(x)
    return c

def findFrequencyOfFrequencyCount(ngram_dict, k, n, V, token_len):
    nc_dict = {}
    nc_dict[0] = (V**n) - token_len

    #n-gram
    for key in ngram_dict:
        if ngram_dict[key] <= k + 1:
            if ngram_dict[key] not in nc_dict:
                nc_dict[ ngram_dict[key]] = 1
            else:
                nc_dict[ ngram_dict[key] ] += 1
    
    #check if all the values of Nc are there in the nc_dict or not ,if there then return           
    val_present = True
    for i in range(1,7):
        if i not in nc_dict:
            val_present = False
            break
    if val_present == True:
        return nc_dict
    
    data_pts, i = {}, 0

    for key in ngram_dict:
        if ngram_dict[key] not in data_pts:
                data_pts[ ngram_dict[key] ] = 1
                i += 1
        if i >5:
            break
            
    for key in ngram_dict:
        if ngram_dict[key] in data_pts:
            data_pts[ ngram_dict[key] ] += 1
    
    #make x ,y coordinates for regression 
    x_coor = [ np.log(item) for item in data_pts ]
    y_coor = [ np.log( data_pts[item] ) for item in data_pts ]
    x = np.array(x_coor, dtype = np.float64)
    y = np.array(y_coor , dtype = np.float64)
   

    #Regression
    slope_m = findBestFitSlope(x,y)
    intercept_c = findBestFitIntercept(x,y,slope_m)

    for i in range(1,(k+2)):
        if i not in nc_dict:
            nc_dict[i] = (slope_m*i) + intercept_c
    
    return nc_dict

def findGoodTuringAdjustCount(c, k, nc_dict):
   
    adjust_count = ( ( (( c + 1)*( nc_dict[c + 1] / nc_dict[c])) - ( c * (k+1) * nc_dict[k+1] / nc_dict[1]) ) /
                     ( 1 - (( k + 1)*nc_dict[k + 1] / nc_dict[1]) )
                   )
    return adjust_count

def doPredictionBackoffGT(input_sen, bi_dict, tri_dict, quad_dict, bi_prob_dict, tri_prob_dict, quad_prob_dict):
    token = input_sen.split()
    
    if input_sen in quad_prob_dict and quad_prob_dict[ input_sen ][0][0]>0:
        pred = quad_prob_dict[input_sen][0]
    elif ' '.join(token[1:]) in tri_prob_dict and tri_prob_dict[' '.join(token[1:])][0][0]>0:
        pred = tri_prob_dict[ ' '.join(token[1:]) ][0]
    elif ' '.join(token[2:]) in bi_prob_dict and bi_prob_dict[ ' '.join(token[2:]) ][0][0]>0:
        pred = bi_prob_dict[' '.join(token[2:])][0]
    else:
        pred = []
    return pred

# Training
def trainCorpus(train_file,test_file,bi_dict,tri_dict,quad_dict,vocab_dict,prob_dict):
      
    test_result, score = '', 0 
    token_len = loadCorpus(train_file, bi_dict, tri_dict, quad_dict, vocab_dict)
    
    k = 5
    V = len(vocab_dict)
    quad_nc_dict = findFrequencyOfFrequencyCount(quad_dict, k, 4, V, len(quad_dict))
    tri_nc_dict = findFrequencyOfFrequencyCount(tri_dict, k, 3, V, len(tri_dict))
    bi_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 2, V, len(bi_dict))
    uni_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 1, V, len(vocab_dict))

    findQuadgramProbGT(vocab_dict, bi_dict, tri_dict, quad_dict, quad_prob_dict, quad_nc_dict, k)
    findTrigramProbGT(vocab_dict, bi_dict, tri_dict, tri_prob_dict, tri_nc_dict, k)
    findBigramProbGT(vocab_dict, bi_dict, bi_prob_dict, bi_nc_dict, k)
    
    sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict)
    
    #TESTING
    test_data = ''

    file = open('corpus/pride-austen.txt','r',encoding="utf8")
    test_data = file.read()

    test_data = removePunctuations(test_data)
    test_token = test_data.split()

    test_token = test_data.split()
    test_quadgrams = list(ngrams(test_token,4))
    
    #BackOff
    score = computeTestScore(test_quadgrams, bi_dict, tri_dict, quad_dict,quad_prob_dict, tri_prob_dict,bi_prob_dict )
    
    #BackOff
    perplexity = computePerplexity(test_quadgrams, bi_dict, tri_dict, quad_dict,vocab_dict,token_len,  k, quad_nc_dict, tri_nc_dict,bi_nc_dict, uni_nc_dict)
    

# Initializing And Main Files

vocab_dict = defaultdict(int)  
bi_dict = defaultdict(int)
tri_dict = defaultdict(int)
quad_dict = defaultdict(int)      

quad_prob_dict = OrderedDict()              
tri_prob_dict = OrderedDict()
bi_prob_dict = OrderedDict()

def getScoresJson(sentence):

    train_file = 'corpus/pride-austen.txt'
    test_file = 'corpus/pride-austen.txt'
    token_len = trainCorpus(train_file,test_file,bi_dict,tri_dict,quad_dict,vocab_dict,quad_prob_dict)

    input_sen = processInput(sentence)
    if input_sen != "invalid":
        word_choice = chooseWords(input_sen, bi_prob_dict, tri_prob_dict, quad_prob_dict)
        #BackOff
        prediction = doPredictionBackoffGT(input_sen, bi_dict, tri_dict, quad_dict, bi_prob_dict, tri_prob_dict, quad_prob_dict)
        final_pred = "null"
        if prediction:
            final_pred = prediction[1]
        else:
            final_pred = "null"
        
        f = open('server-log-outputs/Good_Turing_Backoff_Score.txt','r',encoding="utf-8")
        totalPredictions = f.readline().split(":")[1]
        correct = f.readline().split(":")[1]
        wrong = f.readline().split(":")[1]
        accuracy = f.readline().split(":")[1]
        perplexity = f.readline().split(":")[1]

        data = {"totalPredictions":totalPredictions.strip(),"correct":correct.strip(),"wrong":wrong.strip(),"accuracy":accuracy.strip(),"perplexity":perplexity.strip(),"prediction":prediction[1]}
        json_data = json.dumps(data)
        print(json_data)
    else:
        data = {"prediction":"null"}
        json_data = json.dumps(data)
        print(json_data)
    
if __name__ == '__main__':
    getScoresJson(sys.argv[2])