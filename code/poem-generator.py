#importing necessary libraries
import dynet as dy
import json
import numpy as np
import time
import collections
import random
import math

class PoemGenerator():
    
    def __init__(self, folderpath, embed_dim, hidden_units, batch_size, num_of_poems = None):
        self.unigrams, self.bigrams, self.V = self.readFile(folderpath, num_of_poems) #creating unigram and bigram dicts
        self.vocabulary = list(self.unigrams.keys()) #vocabulary list
        self.training_data, self.embeddings = self.arrangeTrainingData(embed_dim) #encoded training data and embedding matrix
        self.minibathces = self.create_mini_batches(batch_size) #creating mini-batches
        dy.renew_cg()
        self.m = dy.ParameterCollection()
        self.C = self.m.add_lookup_parameters((embed_dim, self.V), init = self.embeddings) #embeddings
        self.U = self.m.add_parameters((hidden_units, embed_dim)) #weights between the embedding layer and the hidden layer
        self.b = self.m.add_parameters((hidden_units)) #biases of the hidden layer
        self.H = self.m.add_parameters((self.V, hidden_units)) #weights between the hidden layer and the output layer
        self.d = self.m.add_parameters((self.V)) #biases of the output layer
        self.trainer = dy.AdamTrainer(self.m) #adam optimizer
        
    def readFile(self, folderpath, num_of_poems = None):
        #reads the given number of poems from the given folderpath, returns unigram and bigram dicts
        unigrams = {}
        bigrams = collections.defaultdict(dict)
        f = open(folderpath)
        data = json.load(f)
        if num_of_poems != None:
            x = num_of_poems
        else:
            x = len(data)
        for d in data[0:x]:
            poem = d['poem'].split('\n')
            for idx in range(len(poem)):
                poem[idx] = '<s> ' + poem[idx].lower() + ' </s>'
                tokens = poem[idx].split()
                for i in range(len(tokens)):
                    unigram_token = tokens[i]
                    try:
                        unigrams[unigram_token] +=1
                    except KeyError:
                        unigrams[unigram_token] = 1
                for i in range(len(tokens)-1):
                    try:
                        inner_dict = bigrams[tokens[i]]
                        try:
                            inner_dict[tokens[i+1]] +=1
                        except KeyError:
                            inner_dict[tokens[i+1]] =1
                    except KeyError:
                        bigrams[tokens[i]][tokens[i+1]] = 1
        f.close()
        V = (len(unigrams))
        return unigrams, bigrams, V
    
    def buildTable(self, dimension): 
        #returns the glove embeddings dictionary
        gloveDict = {}
        with open('glove.6B.{}d.txt'.format(dimension), encoding='utf-8') as f:
            for line in f:
                splitted_line = line.split()
                word = splitted_line[0]
                vector = np.asarray(splitted_line[1:], dtype='float32')
                gloveDict[word] = vector
            f.close()  
        return gloveDict
    
    def arrangeTrainingData(self,dim):
        #returns the encoded training data and the embedding matrix
        training_data = {}
        embeddings = []
        gloveDict = self.buildTable(dim)
        idx=0
        for word in self.vocabulary:
            training_data[word] = []
            x = np.zeros(self.V,)
            x[idx] = 1
            training_data[word].append(x)
            y = self.getTrueOutput(word)
            training_data[word].append(y)
            try:
                embeddings.append(gloveDict[word])
            except:
                embeddings.append(np.random.rand(dim))
            idx += 1
        return training_data, np.asarray(embeddings)

    def getTrueOutput(self, word):
        #returns the encoded truth vector of a given word
        try:
            inner_dict = self.bigrams[word] 
            maxKey = max(inner_dict, key=inner_dict.get)
            idx = self.vocabulary.index(maxKey)
            y = np.zeros(self.V,)
            y[idx] = 1
        except ValueError:
            y = np.zeros(self.V,)
        return y
    
    def create_mini_batches(self,batch_size):
        #shuffles the data and divides it into minibatches 
        keys = list(self.training_data.keys())
        random.shuffle(keys)
        mini_batches = [keys[i: i + batch_size] for i in range(0, len(keys), batch_size)]
        return mini_batches
    
    def train(self, epoch):
        #train process of the neural network
        history = []
        for i in range(epoch):
            start = time.time()
            total_loss = 0
            print('Epoch ' + str(i+1) + ':')
            for batch in self.minibathces:
                dy.renew_cg()
                losses = []
                for word in batch:
                    x = self.training_data[word][0]
                    y = self.training_data[word][1]
                    dy_x = dy.inputVector(x)
                    dy_y = dy.inputVector(y)
                    output = self.feedForward(dy_x)
                    l = self.calculateLoss(output, dy_y)
                    losses.append(l)
                loss = dy.esum(losses) / len(losses)
                total_loss += loss.value()
                loss.backward()
                self.trainer.update()
            end = time.time()
            print('Loss = {0}\nTime it takes = {1} minutes.'.format(total_loss/len(self.minibathces), (end-start)/60))
            history.append(total_loss/len(self.minibathces))
        return history
        
    def feedForward(self, x):
        #returns the output vector of a given input vector
        embedding = (self.C * x)
        hidden_layer = self.getHiddenLayer(embedding)
        output_layer = self.getOutputLayer(hidden_layer)
        return output_layer
        
    def getHiddenLayer(self, embedding):
        #helper function to get hidden layer vector
        return dy.tanh((self.U * embedding) + self.b)
    
    def getOutputLayer(self, hidden_layer):
        #helper function to get output layer vector
        return dy.softmax((self.H * hidden_layer) + self.d)
    
    def calculateLoss(self, output, y):
        #function to calculate the loss value
        loss = dy.binary_log_loss(output, y)
        return loss
    
    def generatePoems(self, num_of_poems, num_of_lines):
        #returns the list of generated poems
        oneHot = self.training_data['<s>'][0]
        poems=[]
        for i in range(num_of_poems):
            poem = []
            for j in range(num_of_lines):
                words = ['<s>']
                dy.renew_cg()
                inputVec = dy.inputVector(oneHot)
                output = self.feedForward(inputVec)
                nextWord, idx = self.getNextWord(output.value())
                words.append(nextWord)
                while(nextWord != '</s>'):
                    dy.renew_cg()
                    next_oneHot = self.training_data[nextWord][0]
                    next_inputVec = dy.inputVector(next_oneHot)
                    output = self.feedForward(next_inputVec)
                    nextWord, idx = self.getNextWord(output.value())
                    words.append(nextWord)
                poem.append(words)
            poems.append(poem)
        return poems
    
    def getNextWord(self, output_layer):
        #using weighted random choice, returns the word from output layer of the network
        r = random.random()
        var = .0
        for i in range(len(output_layer)):
            var += output_layer[i]
            if var >= r:
                break
        word = self.vocabulary[i]
        return word, i
    
    def printPoems(self, poems):
        #prints generated poems
        for i in range(len(poems)):
            print('Poem {}:'.format(i+1))
            for line in poems[i]:
                print(' '.join(line[1:len(line)-1]))
            print('Perplexity for poem {0} = {1}'.format(i+1, self.calcPerplexity(poems[i])))
            print('\n')
        return
    
    def calculateProb(self, poem):
        #calculates and returns the probability of a given poem
        poem_prob = 0
        total_N = 0
        for line in poem:
            line_prob = 0
            N = len(line)
            for i in range(len(line)-1):
                prev_word = line[i]
                next_word = line[i+1]
                try:
                    prob = (self.bigrams[prev_word][next_word] + 1)/(self.unigrams[prev_word] + self.V)
                except:
                    prob = 1/(self.unigrams[prev_word] + self.V)
                line_prob += math.log2(prob)
            total_N += N
            poem_prob += line_prob
        return poem_prob, total_N
    
    def calcPerplexity(self, poem):
        #calculates and returns the perplexity of a given poem
        poem_prob, N = self.calculateProb(poem)
        poem_prob_for_ppl = 2**(poem_prob)
        ppl = poem_prob_for_ppl**(-1/float(N))
        return ppl
    

#object creation   
start = time.time()
model = PoemGenerator('unim_poem.json', embed_dim=50, hidden_units=8, batch_size=4, num_of_poems=100)
end = time.time()
print('Time the program takes to create the model: ' + str((end-start)/60)+ ' minutes.') 

losses = model.train(epoch=5) #training the model

poems = model.generatePoems(num_of_poems= 5, num_of_lines=4) #generating poems
model.printPoems(poems)