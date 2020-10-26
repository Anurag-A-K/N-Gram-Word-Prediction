import nltk as nltk
from nltk.util import ngrams
from collections import defaultdict
from collections import OrderedDict
import string
import json
import sys
import os

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

def loadCorpus(file_path, bi_dict, tri_dict, quad_dict, vocab_dict):

    w1, w2, w3 = '', '', ''       
    word_len , token = 0, []

    file = open(file_path,'r',encoding="utf8")
    for line in file:
        temp_l = line.split()
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
    V,i = len(vocab_dict),0
   
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
        return "invalid"
    else:
        cond = True
        temp = temp[-3:]
        sen = " ".join(temp)
        return sen

#Interpolation
def computeTestScore(test_token, bi_dict, tri_dict, quad_dict, vocab_dict,token_len, param, k, quad_nc_dict, tri_nc_dict, bi_nc_dict, uni_nc_dict ):
    score,wrong,total = 0,0,0
    w = open('server-log-outputs/Good_Turing_Interpolation_Score.txt','w',encoding="utf-8")
    for sent in test_token:
        sen_token = sent[:3]
        sen = " ".join(sen_token)
        correct_word = sent[3]

        #find the the most probable words for the bigram, trigram and unigram  sentence               
        word_choice = chooseWords(sen, bi_prob_dict, tri_prob_dict, quad_prob_dict)

        result = doInterpolatedPredictionGT(sen, bi_dict, tri_dict, quad_dict,vocab_dict,token_len, word_choice, param, k, quad_nc_dict, tri_nc_dict,bi_nc_dict, uni_nc_dict )
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

# General Method to estimate parameters, doesn't work.
# def estimateParameters(token_len, vocab_dict, bi_dict, tri_dict, quad_dict):
#     max_prob = -9999999999999999999.0
#     curr_prob = 0.0
#     parameters = [0.0,0.0,0.0,0.0]
#     i = 1
    
#     #load the held out data 
#     file = open('held_out_corpus.txt','r')
#     held_out_data = file.read()
#     file.close()
#     held_out_data = removePunctuations(held_out_data)
#     held_out_data = held_out_data.split()
#     quad_token_heldout = list(ngrams(held_out_data,4))

#     l1 = l4 = 0
#     while l1 <= 1.0:
#         l2 = 0
#         while l2 <= 1.0:
#             l3 = 0
#             while l3 <= 1.0:
#                 if l1 == 0 and l2 == 0 and l3 == 0 or ((l1+l2+l3)>1):
#                     l3 += 0.1
#                     i += 1
#                     continue
                    
#                 l4 = 1- (l1 + l2 + l3)
                
#                 curr_prob = 0
#                 qc = [0]
#                 bc = [0]
#                 tc = [0]
                
#                 #find the probability for the held out set using the current lambda values
#                 for quad in quad_token_heldout:
#                     curr_prob += log10( interpolatedProbability(quad,token_len, vocab_dict, bi_dict, tri_dict, quad_dict,qc,bc,tc,l1, l2, l3, l4) )
#                 if curr_prob > max_prob:
#                     max_prob = curr_prob
#                     parameters[0] = l1
#                     parameters[1] = l2
#                     parameters[2] = l3
#                     parameters[3] = l4
#                 l3 += 0.1
#                 i += 1
#             l2 += 0.1
#         l1 += 0.1
#     return parameters

def doInterpolatedPredictionGT(sen, bi_dict, tri_dict, quad_dict,vocab_dict,token_len, word_choice, param, k, quad_nc_dict, tri_nc_dict,bi_nc_dict, uni_nc_dict ):
    pred, max_prob = '',0.0
    V = len(vocab_dict)

    for word in word_choice:
        key = sen + ' ' + word[1]
        quad_token = key.split()
        
        quad_count = quad_dict[key]
        tri_count = tri_dict[' '.join(quad_token[0:3])]
        
        if quad_dict[key] <= k or (key not in quad_dict):
            quad_count = findGoodTuringAdjustCount( quad_dict[key], k, quad_nc_dict)
        if tri_dict[' '.join(quad_token[0:3])] <= k  or (' '.join(quad_token[0:3]) not in tri_dict):
            tri_count = findGoodTuringAdjustCount( tri_dict[' '.join(quad_token[0:3])], k, tri_nc_dict)
        quad_prob = quad_count / tri_count
        
        tri_count = tri_dict[' '.join(quad_token[1:4])]
        bi_count = bi_dict[' '.join(quad_token[1:3])]
        
        if tri_dict[' '.join(quad_token[1:4])] <= k  or (' '.join(quad_token[1:4]) not in tri_dict):
            tri_count = findGoodTuringAdjustCount( tri_dict[' '.join(quad_token[1:4])], k, tri_nc_dict)
        if bi_dict[' '.join(quad_token[1:3])] <= k or (' '.join(quad_token[1:3]) not in bi_dict):
            bi_count = findGoodTuringAdjustCount( bi_dict[' '.join(quad_token[1:3])], k, bi_nc_dict)
        tri_prob = tri_count / bi_count
       
        bi_count = bi_dict[' '.join(quad_token[2:4])]
        uni_count = vocab_dict[quad_token[2]]
        
        if bi_dict[' '.join(quad_token[2:4])] <= k or (' '.join(quad_token[2:4]) not in bi_dict):
            bi_count = findGoodTuringAdjustCount( bi_dict[' '.join(quad_token[2:4])], k, bi_nc_dict)
        if vocab_dict[quad_token[2]] <= k or (quad_token[2] not in vocab_dict):
            uni_count = findGoodTuringAdjustCount( vocab_dict[quad_token[2]], k, uni_nc_dict)
        bi_prob = bi_count / uni_count
        
        uni_count = vocab_dict[quad_token[3]]
        
        if vocab_dict[quad_token[3]] <= k or (quad_token[3] not in vocab_dict):
            bi_count = findGoodTuringAdjustCount( vocab_dict[quad_token[3]], k, uni_nc_dict)
        uni_prob = uni_count / token_len
        
        prob = (   
                  param[0]*( quad_prob ) 
                + param[1]*( tri_prob ) 
                + param[2]*( bi_prob ) 
                + param[3]*(uni_prob)
               )
       
        if prob > max_prob:
            max_prob = prob
            pred = word

    if pred:
        return pred
    else:
        return ''

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

#Interpolation
def computePerplexity(test_quadgrams, bi_dict, tri_dict, quad_dict,vocab_dict,token_len, param, k, quad_nc_dict, tri_nc_dict,bi_nc_dict, uni_nc_dict):
    
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

    w = open('server-log-outputs/Good_Turing_Interpolation_Score.txt','a',encoding="utf-8")
    w.write('\nPerplexity: '+str(perplexity))
    
    return perplexity

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style

def findBestFitSlope(x,y):
    m = (( mean(x)*mean(y) - mean(x*y) ) / 
          ( mean(x)** 2 - mean(x**2)))

    return m
      
#finds the intercept for the best fit line
def findBestFitIntercept(x,y,m):
    c = mean(y) - m*mean(x)
    return c

def findFrequencyOfFrequencyCount(ngram_dict, k, n, V, token_len):
    nc_dict = {}
    nc_dict[0] = V**n - token_len

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
    
    data_pts = {}
    i = 0

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


def trainCorpus(train_file,test_file,bi_dict,tri_dict,quad_dict,vocab_dict,prob_dict):
      
    test_result = ''
    score = 0
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
    param = [0.3,0.1,0.1,0.5]
        
    #TESTING
    test_data = ''

    file = open('corpus/pride-austen.txt','r',encoding="utf8")
    test_data = file.read()

    test_data = removePunctuations(test_data)
    test_token = test_data.split()

    test_token = test_data.split()
    test_quadgrams = list(ngrams(test_token,4))
    
    #Interpolation
    score = computeTestScore(test_quadgrams, bi_dict, tri_dict, quad_dict,vocab_dict,token_len, param, k, quad_nc_dict, tri_nc_dict,bi_nc_dict, uni_nc_dict )
    #Interpolation
    perplexity = computePerplexity(test_quadgrams, bi_dict, tri_dict, quad_dict,vocab_dict,token_len, param, k, quad_nc_dict, tri_nc_dict,bi_nc_dict, uni_nc_dict)  


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

    param = [0.7,0.1,0.1,0.1]
    
    input_sen = processInput(sentence)
    word_choice = chooseWords(input_sen, bi_prob_dict, tri_prob_dict, quad_prob_dict)
    if input_sen != "invalid":
        word_choice = chooseWords(input_sen, bi_prob_dict, tri_prob_dict, quad_prob_dict)
        #Interpolation
        prediction = doInterpolatedPredictionGT(input_sen, bi_dict, tri_dict, quad_dict,vocab_dict,token_len, word_choice, param, k, quad_nc_dict, tri_nc_dict,bi_nc_dict, uni_nc_dict )
        final_pred = "null"
        if prediction:
            final_pred = prediction[1]
        else:
            final_pred = "null"
    
        f = open('server-log-outputs/Good_Turing_Interpolation_Score.txt','r',encoding="utf-8")
        totalPredictions = f.readline().split(":")[1]
        correct = f.readline().split(":")[1]
        wrong = f.readline().split(":")[1]
        accuracy = f.readline().split(":")[1]
        perplexity = f.readline().split(":")[1]

        data = {"totalPredictions":totalPredictions.strip(),"correct":correct.strip(),"wrong":wrong.strip(),"accuracy":accuracy.strip(),"perplexity":perplexity.strip(),"prediction":final_pred}
        json_data = json.dumps(data)
    else:
        data = {"prediction":"null"}
        json_data = json.dumps(data)

    jF = open('output.json','r+',encoding="utf-8")
    jF.truncate()
    jF.close()
    jF2 = open('output.json','w',encoding="utf-8")
    jF2.write(json_data)
    jF2.close()
    print(json_data)

if __name__ == '__main__':
    getScoresJson(sys.argv[2])