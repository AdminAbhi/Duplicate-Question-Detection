import numpy as np
import pandas as pd
import nltk
import random as rnd
import pickle

#!pip install trax
nltk.download('punkt')
from trax import layers as tl
from trax.fastmath import numpy as fastnp

from collections import defaultdict

voc = pickle.load(open('vocab.pkl', 'rb'))

vocab = defaultdict(lambda: 0)

for i in voc:
  vocab[i] = voc[i]

print('The length of the vocabulary is: ', len(vocab))


def data_generator(Q1, Q2, batch_size, pad=1, shuffle=True):
    input1 = []
    input2 = []
    idx = 0
    len_q = len(Q1)
    question_indexes = [*range(len_q)]
    
    if shuffle:
        rnd.shuffle(question_indexes)
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    while True:
        if idx >= len_q:
            # if idx is greater than or equal to len_q, set idx accordingly 
            # (Hint: look at the instructions above)
            idx = len_q
            # shuffle to get random batches if shuffle is set to True
            if shuffle:
                rnd.shuffle(question_indexes)
        
        # get questions at the `question_indexes[idx]` position in Q1 and Q2
        q1 = Q1[question_indexes[idx]]
        q2 = Q2[question_indexes[idx]]
        
        # increment idx by 1
        idx += 1
        # append q1
        input1.append(q1)
        # append q2
        input2.append(q2)
        if len(input1) == batch_size:
            # determine max_len as the longest question in input1 & input 2
            # Hint: use the `max` function. 
            # take max of input1 & input2 and then max out of the two of them.
            max_len = max(max([len(q) for q in input1]),max([len(q) for q in input2]))
            # pad to power-of-2 (Hint: look at the instructions above)
            max_len = 2**int(np.ceil(np.log2(max_len)))
            b1 = []
            b2 = []
            for q1, q2 in zip(input1, input2):
                # add [pad] to q1 until it reaches max_len
                q1 = q1 + [pad] * (max_len - len(q1))
                # add [pad] to q2 until it reaches max_len
                q2 = q2 + [pad] * (max_len - len(q2))
                # append q1
                b1.append(q1)
                # append q2
                b2.append(q2)
            # use b1 and b2
            yield np.array(b1), np.array(b2)
    ### END CODE HERE ###
            # reset the batches
            input1, input2 = [], []  # reset the batches


def Siamese(vocab_size=len(vocab), d_model=128, mode='train'):

    def normalize(x):  # normalizes the vectors to have L2 norm 1
        return x / fastnp.sqrt(fastnp.sum(x * x, axis=-1, keepdims=True))
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    q_processor = tl.Serial(  # Processor will run on Q1 and Q2.
        tl.Embedding(vocab_size, d_model), # Embedding layer
        tl.LSTM(d_model), # LSTM layer
        tl.Mean(axis=1), # Mean over columns
        tl.Fn('Normalize', lambda x: normalize(x))  # Apply normalize function
    )  # Returns one vector of shape [batch_size, d_model].
    
    ### END CODE HERE ###
    
    # Run on Q1 and Q2 in parallel.
    model = tl.Parallel(q_processor, q_processor)
    return model


def TripletLossFn(v1, v2, margin=0.25):
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    
    # use fastnp to take the dot product of the two batches (don't forget to transpose the second argument)
    scores = fastnp.dot(v1, v2.T)  # pairwise cosine sim
    # calculate new batch size
    batch_size = len(scores)
    # use fastnp to grab all postive `diagonal` entries in `scores`
    positive = fastnp.diagonal(scores)  # the positive ones (duplicates)
    # multiply `fastnp.eye(batch_size)` with 2.0 and subtract it out of `scores`
    negative_without_positive = scores - 2.0 * fastnp.eye(batch_size)
    # take the row by row `max` of `negative_without_positive`. 
    # Hint: negative_without_positive.max(axis = [?])  
    closest_negative = negative_without_positive.max(axis=1)
    # subtract `fastnp.eye(batch_size)` out of 1.0 and do element-wise multiplication with `scores`
    negative_zero_on_duplicate = scores * (1.0 - fastnp.eye(batch_size))
    # use `fastnp.sum` on `negative_zero_on_duplicate` for `axis=1` and divide it by `(batch_size - 1)` 
    mean_negative = np.sum(negative_zero_on_duplicate, axis=1) / (batch_size-1)
    # compute `fastnp.maximum` among 0.0 and `A`
    # A = subtract `positive` from `margin` and add `closest_negative` 
    triplet_loss1 = fastnp.maximum(0.0, margin - positive + closest_negative)
    # compute `fastnp.maximum` among 0.0 and `B`
    # B = subtract `positive` from `margin` and add `mean_negative`
    triplet_loss2 = fastnp.maximum(0.0, margin - positive + mean_negative)
    # add the two losses together and take the `fastnp.mean` of it
    triplet_loss = fastnp.mean(triplet_loss1 + triplet_loss2)
    
    ### END CODE HERE ###
    
    return triplet_loss


def get_predict(question1, question2, threshold, model, vocab, data_generator=data_generator, verbose=False):

    ### START CODE HERE (Replace instances of 'None' with your code) ###
    # use `nltk` word tokenize function to tokenize
    q1 = nltk.word_tokenize(question1)  # tokenize
    q2 = nltk.word_tokenize(question2)  # tokenize
    Q1, Q2 = [], []
    for word in q1:  # encode q1
        # increment by checking the 'word' index in `vocab`
        Q1 += [vocab[word.lower()]]
    for word in q2:  # encode q2
        # increment by checking the 'word' index in `vocab`
        Q2 += [vocab[word.lower()]]
        
    # Call the data generator (built in Ex 01) using next()
    # pass [Q1] & [Q2] as Q1 & Q2 arguments of the data generator. Set batch size as 1
    # Hint: use `vocab['<PAD>']` for the `pad` argument of the data generator
    Q1, Q2 = next(data_generator([Q1], [Q2], 1, vocab['<PAD>']))
    # Call the model
    v1, v2 = model((Q1, Q2))
    # take dot product to compute cos similarity of each pair of entries, v1, v2
    # don't forget to transpose the second argument
    d = np.dot(v1[0], v2[0].T)
    # is d greater than the threshold?
    res = d > threshold
    
    ### END CODE HERE ###
    
    if(verbose):
        print("Q1  = ", Q1, "\nQ2  = ", Q2)
        print("d   = ", d)
        print("res = ", res)

    return [res, d]


model = Siamese()
model.init_from_file('model.pkl.gz')


