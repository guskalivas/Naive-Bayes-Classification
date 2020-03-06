# Name: Gus Kalivas

import os
import math

'''
Train takes in a directory and a cutoff and creates a vocab list, training data set, calculates
the prior for years 2020 and 2016, and returns a dic of the vocab, log prior, log probability
of a word give a year 
'''
def train(training_directory, cutoff):
    train = {} # empty dic to fill
    vocab = create_vocabulary(training_directory,cutoff)
    # creates training data from vocab and 
    training_data = load_training_data(vocab, training_directory)
    p = prior(training_data, ['2020', '2016'])
    # create dictionary to return of all the info
    train['vocabulary'] = vocab
    train['log prior'] = p
    train['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')
    train['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')
    return train

'''
create vocabulary takes in a training directory and cut off and creates a list of vocab words in 
the directory that appear more then the cut off
'''
def create_vocabulary(training_directory, cutoff):
    # joins the paths of trianing and years to loop through all files in those subdirectories
    files = os.path.join(training_directory, 'training')
    f2016 = os.path.join(files, '2016')
    f2020 = os.path.join(files, '2020')
    dic = {}
    vocab = []
    # loop through 2016 files 
    for i in os.listdir(f2016):
        # open each file 
        with open(os.path.join(f2016, i),'r', encoding='utf-8') as f:
            # for each word in f 
            for j in f:
                # remove any new lines 
                j = j.replace('\n', '')
                # if not in the dictonary already set equal to one 
                if j not in dic:
                    dic[j] = 1
                else: # otherwise increment count 
                    dic[j] +=1
    # loop and do same thing with 2020 files 
    for i in os.listdir(f2020):
        with open(os.path.join(f2020, i), 'r', encoding='utf-8') as f:
            for j in f:
                j = j.replace('\n', '')
                if j not in dic:
                    dic[j] = 1
                else:
                    dic[j] +=1
    # for each word and value in the dictonary if vlaue is greater then cut off keep it 
    dic = {key:value for key, value in dic.items() if value >= cutoff}
    dic = sorted(dic) 
    return dic # return a sorted list 
    
'''
create bow creates a bag of words for a given file given a set of vocab
'''
def create_bow(vocab, filepath):
    bow = {}
    count = 0 
    # open the file 
    with open(filepath, encoding= 'utf-8') as f:
        # loop and replace any new line characters 
        for i in f:
            i = i.replace('\n', '')
            # if this word is in the vocab 
            if i in vocab:
                if i not in bow: # if not in bow yet set equal to one 
                    bow[i] = 1
                else: # else increment the value of this word 
                    bow[i] +=1
            else: # else count this word as OOV
                count +=1
    if count > 0: # if count > 0 then theres OOV in this BOW
        result = bow
        result[None] = count # place OOV value with key None in dictonary
        return result # return it 
    return bow #return the bow dictonary 

'''
load training data takes in vocab and directory and returns a dictionary with each file
and its associated label year and its BOW
'''
def load_training_data(vocab, directory):
    # append on training and the years to each file path
    train = os.path.join(directory, 'training') 
    full2016 = os.path.join(train, '2016')
    full2020 = os.path.join(train, '2020')
    train_data = []
    # loop through the 2016 files 
    for i in os.listdir(full2016):
        d = {} # create its label and bow 
        d['label'] = '2016'
        d['bow'] = create_bow(vocab, os.path.join(full2016, i))
        train_data.append(d) # append to full list 
    # same thing for list of 2020 files 
    for j in os.listdir(full2020):
        d = {}
        d['label'] = '2020'
        d['bow'] = create_bow(vocab, os.path.join(full2020, j))
        train_data.append(d)
    return train_data # reutrn the full list 

'''
prior takes in the training data and label list of years 
'''
def prior(training_data, label_list):
    prior = {} 
    count1 = 0
    count2 = 0
    # loop through training data 
    for i in training_data:
        #count the number of labels with 2016
        if i['label'] == label_list[0]:
            count1 += 1
        # count the number of labels with 2020
        if i['label'] == label_list[1]:
            count2 += 1
    # divide the num of 2016/2020 files over the length of trianing data 
    p1 = count1/len(training_data)
    p2 = count2/len(training_data)
    # take the log probability of each value 
    prior[label_list[0]] = math.log(p1)
    prior[label_list[1]] = math.log(p2)

    return prior # return the dictionary 

'''
p_word_given_label takes in a set of vocabulary, training data and label to calculate the 
log probability of a word given its label 
'''
def p_word_given_label(vocab, training_data, label):
    d = {} 
    count1 = 0
    w = {}
    # loop through the training data 
    for i in training_data:
        # for each word value in the first files bow
        for word, value in i['bow'].items():
            if i['label'] == label: 
                # if the word is in the vocab 
                if word in vocab:
                    # and not in the new dic yet
                    if word not in d:
                        # set the value to its value in the bow in the new dic 
                        d[word] = value
                    else:
                        # else increment and add all the existing value with this value
                        d[word] += value
                else: # else count the values of OOV words
                    count1+= value
    # loop through the vocab and dictionary 
    for j in vocab:
        for x in d:
            # if a word in vocab is not in the dictionary 
            if j not in d:
                # set its value to 0 
                w[j] = 0
            else: # else set its value to the value in the dic we just created 
                w[j] = d[j]
    w[None] = count1 # set None equal to OOV values 
    return calcProb(w) # return helper method to calc log probailities 

'''
helper method takes in the full list of words and their values 
'''
def calcProb(all_words):
    final = {}
    l = len(all_words) # get the length of the dic
    values = sum(all_words.values()) # sum all the values in the dic 
    dem = values + l # demoniator for equal is values plus the length 
    # loop through all the words in the dic 
    for word, value in all_words.items():
        # calculate the numerator word value + 1
        num = (value + 1)
        # set its value as the log of num - log of dem 
        final[word] = (math.log(num) - math.log(dem))
   
    return final # return the list 

'''
classify takes in a trained model and a file path to a specific file to classify
'''
def classify(model, filepath):
    # create a bow given the models vocab and the file path 
    bow = create_bow(model['vocabulary'], filepath)
    st16 = 0
    st20 = 0
    # for word value in the log prior dic in the model for 2016
    for x, y in model['log p(w|y=2016)'].items():
        # for each word value in the bow 
        for word, value in bow.items():
            # if the words match add the log prior times the number of words 
            if x == word:
                st16+= y*value
    # add the prior for year 2016
    st16 += model['log prior']['2016']
    # same thing for log of a word given 2020 
    for x, y in model['log p(w|y=2020)'].items():
        for word, value in bow.items():
            if x == word:
                st20+=y*value
    st20 += model['log prior']['2020']
    d = {}
    ret = ''
    # if 2020 is greater then 2016 value 
    if st20 > st16:
        ret = '2020'
    else:
        ret = '2016'
    # set the return value to the larger of the two 
    d['log p(y=2020|x)'] = st20
    d['log p(y=2016|x)'] = st16
    d['predicted y:'] = ret
    return d # return values for each year and predicted year 


vocab = create_vocabulary('corpus/corpus', 2000)
#bow = create_bow(vocab,'corpus/training/corpus/training/2016/1.txt' )
load = load_training_data(vocab, './corpus/training/corpus/')
p = prior(load, ['2020', '2016'])
print(p)
#model = train('corpus/corpus/', 2)
#c = classify(model, 'corpus/corpus/test/2016/0.txt')
#print(c)
