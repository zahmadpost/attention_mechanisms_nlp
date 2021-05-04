# Attention Mechanisms in Deep Learning Natural Language Processing Tasks

This tutorial is an introduction to attention mechanisms. It will walk you through implementing additive attention mechanisms 'from scratch' using numpy. The additive attention mechanism is introduced in a deep learning-based natural language processing model that can be used in information retrieval tasks. 

## Introduction
In the last decade, many advancements in natural language processing (NLP) have come from the domain of deep learning (DL). DL architectures such as deep neural networks (DNNs), convolutional neural networks (CNNs), recurrent neural networks (RNNs), and long-short term memory networks (LSTMs) have revolutionized text representation and text processing. Newer architectures, such as transformers, and encoders/decoders have made dramatic improvements in machine translation tasks. Attention mechanisms are closely related to transformers, and have demonstrated their usefulness not only in NLP tasks, but image recognition tasks as well. 

## What is an Attention Mechanism?
An attention mechanism, whether as a standalone architecture, or as a layer in a larger model, allows one to understand how words in a sentence (or in pairs of sentences) relate to one another. Unlike LSTMs, they are size-agnostic: they can identify relationships between words both close and far away from the target word. Attention mechanisms help solve issues that arise from more traditional word vectors as well. Word vectors often encode the same representation for a word that has many semantic meanings. Attention mechanisms can calculate the relationship of each word in a text to every other word in the text (this process is called self-attention), or to every other word in a related text. These calculations express a word’s meaning in a context vector. The goal is to quantify the contextual meaning of any given word in a sentence and determine which words supply the greatest semantic relevance with respect to the word in question. 

## How does it Operate?
Attention mechanisms use key-value-query schemas to learn context vectors. Given some input text, a word in that text may act as a query term. The query will search through all keys of all words that could supply contextual meaning for the query term. The keys each have values that encode meaning about the key term. The relationships between queries and keys, and keys and values are trainable: the attention mechanism learns these relationships as it trains. The output of this training is a vector of importance weights. We can use these weights to estimate how strongly correlated words are to other elements in a sentence. The key take-away is that attention mechanisms help us understand which parts of a text or image to pay attention to, rather than weighting each part of the text or image equally. 

## How Do I Implement an Attention Mechanism?
Due to their soaring popularity in recent years, attention mechanisms have been implemented in Keras, a Python library popular in machine learning and artificial intelligence communities. Using the powerful Keras API, a researcher can implement the dot-product attention mechanism in one or two lines of code. However, Keras does not support other kinds of attention mechanisms (e.g. additive, content-based, location-based, scaled dot-product). This tutorial will walk through how to implement an additive attention mechanism in a deep learning model. Although we utilize the Keras functional API, we will not use the Keras attention layer. Instead, we will walk through building an additive attention mechanism from scratch using numpy.

## Information Retrieval
Information Retrieval (IR) is one family of tasks within NLP. Within IR are several common tasks: question answering, text matching, search engines, and recommender systems. The critical goal of IR is to reduce information overload, that is, to provide the most relevant information from a large collection of resources. As attention mechanisms seek to understand exactly which words in a sentence or sentences in a text are the most important in their context, they may be well suited for IR tasks. 

## MatchZoo
In this tutorial, we will understand not only how additive attention mechanisms function, we will also use one in the context of IR. Specifically, we will use an additive attention mechanism within a DL model designed for short text matching. Short text matching uses an NLP model to predict the semantic relevance of two texts (citation: https://www.cil.pku.edu.cn/docs/2019-07/20190731123828506844.pdf) In order to set up the structure of our short text matching model, we will be utilizing MatchZoo, a Python library that implements several state-of-the-art DL-based short text matching models (citation: https://github.com/NTMC-Community/MatchZoo ) including DSSM, MV-LSTM, and Conv-KNRM. MatchZoo provides a convenient data object called a data pack, which takes as inputs three columns of text. The first column contains the query text. In IR, a query text is the text that we know, or wish to know more about. The second column contains the document text. Document texts can be any number of texts from an information store. The document text is what we need to filter through to find a semantically relevant match to the query text. As we are training a supervised DL model, the third column will be the labels. These labels tell our model which query-document text pairs are relevant and which are not. MatchZoo’s data pack object stores this information for us and preempts a good deal of pandas data frame manipulations while preprocessing the text. 

***!!! Important: MatchZoo relies on an older version of Keras.metrics. It would be a good idea (and may be necessary) to create a new virtual environment (venv) before installing MatchZoo and running your code!!!***

## Defining our NLP/IR Task

Before we dig into the code, let’s define a toy problem: say we are searching for Tweets that contain news article headlines. We want to know how many Tweets contain messaging similar to a given headline. Perhaps we are trying to trace the spread of a story on Twitter but can’t rely exclusively on retweets or exact copies of Tweet bodies. We want to know how far the idea of a story has gone on Twitter. We first start with the article headline text. Then we collect a sampling of tweets within a given timeframe and collect the body text. We label the relevant Tweet as a match. To train our model, we will need to provide tens or hundreds of examples like the one below:

|Query Headline | Document Headline | Match |
|-------------- | ------------------| -------|
|Ivy League cancels winter sports because of COVID-19 | College football games canceled or postponed because of the coronavirus | 1 |
|Ivy League cancels winter sports because of COVID-19 | A House Divided | 0 |
|Ivy League cancels winter sports because of COVID-19| Tillis Ekes Out Victory in North Carolina, Bolstering Republicans’ Hold on the Senate | 0 |
|Ivy League cancels winter sports because of COVID-19 | Kamala Harris Avoids Saying 'Net Zero' in Texas, Pennsylvania, But Says It in Other States | 0 |
|Ivy League cancels winter sports because of COVID-19 | Lions' Matthew Stafford on biggest struggles this season: 'Um, all of it' | 0 |

We can see that the query and document headlines have very few words in common. Although they are semantically similar, a traditional text matching approach would have a great deal of trouble matching these headlines. We need to understand which words in the query help to explain the words in the document. This is where our attention mechanism comes in!
Let’s build a model that can help us determine which of the document texts is a semantic match to the query text. ***All of the code below can be found in a jupyter lab notebook in this repository. Additionally, some test data is available as a csv file.***
Fist, we’ll use the MatchZoo library to create a custom preprocessor. First,  we use Word2Vec (citation) trained on the Google News 300 data set to create 300-dimensional vectors to represent each word in each sentence of the query and document texts. The resulting embeddings for each document and each query will be a 30 by 300-dimensional matrix. We use 30 because most article headlines are generally shorter than 30. We will create symmetric padding (e.g. place rows of 0’s to the left and to the right) around each word in the matrix until it fills the 30x 300 shape. 

###Preprocessor:
First, import the required packages
```
from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from nltk.tokenize import RegexpTokenizer
import gensim.downloader as gensim
import tensorflow as tf
from matchzoo.preprocessors import units
import numpy as np
from matchzoo.preprocessors.units.unit import Unit
import matchzoo as mz
```
Next, set up the word embedding model
```
wordembeddingmodel = gensim.load('word2vec-google-news-300')
```

Create symmetrical padding for each text input
```
class FixedLengthUnit(Unit):
    def __init__(self, fixed_length: int):
        self._fixed_length = fixed_length
   
    def transform(self, input_: tf.Tensor) -> tf.Tensor:
        input_length = input_.shape[1]
        if input_length == 0:
            input_ = tf.zeros([300,0])
        pre_ct = (self._fixed_length-input_length)//2
        post_ct = self._fixed_length-pre_ct-input_length
        pre = tf.zeros([300,pre_ct])
        post = tf.zeros([300,post_ct])
        return tf.concat([pre, input_, post], axis=1)
```
Create the Attention Model Preprocessor (superclass is Base Preprocessor)
```
class AttentionModelPreprocessor(BasePreprocessor):
    def __init__(self, fixed_length: int):
        super().__init__()
        self._fixed_length= fixed_length
        self._fixedlength_unit = FixedLengthUnit(self._fixed_length)
        self._tokenizer = RegexpTokenizer('[^\d\W]+')
#Filter the vocabulary to only include words in the google news 300 dataset
    def _filter_vocab(self, input_: list) ->list:
        return list(word for word in input_ if word in wordembeddingmodel.vocab)
#Create the word embeddings for each word in the article headline
    def _process(self, input_: list) ->list:
         #get word embeddings
         vectors = list(wordembeddingmodel[word] for word in input_)
#Create fit and transform methods (the fit method doesn’t actually do anything but is required by the MatchZoo API)
    def fit(self, data_pack: DataPack, verbose: int=1):
        return self
#Use the MatchZoo data_pack objects to apply the tokenization on each side of the datapack (query and document texts)
    def transform(self, data_pack: DataPack, verbose: int=1) -> DataPack:
        data_pack = data_pack.copy()
        data_pack.apply_on_text(self._tokenizer.tokenize, inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._filter_vocab, inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._process, inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._fixedlength_unit.transform, inplace=True, verbose=verbose)
        return data_pack
```
Next, we’ll create the model using the Keras API. We’ll define the attention layer and weights:
```
class AttentionTextMatchingModel():
     
    def build(self):
```
We start by specifying the input dimensions. In this case, the input dimensions are the batchsize, 30 (for the fixed length unit defined in the preprocessor) and 300 (representing the 300-d vectors generated by Word2Vec in the preprocessor.) Note that in this method, the term ***query*** means the document text and ***value*** is the query text.
```
    def select_weight(docs):
            #both inputs have same size: [batchsize, 30, 300]
            query, value = docs
```
We then reshape the query to be [batchsize, 30, 1, 300] and the value to be [batchsize, 1, 30, 300]. This is a preparatory step that will allow for broadcasting. 
```
    #reshape inputs
            #query reshaped size is [batchsize, 30, 1, 300]
            query_reshaped = tf.expand_dims(query, axis=2)
            #value reshaped size is [batchsize, 1, 30, 300]
            value_reshaped = tf.expand_dims(value, axis=1)
```
We’ve broadcast dimension 1 (0-indexed) of the value and dimension 2 of the query (0-indexed). Now we can add these tensors together using the tensorflow add function.  
```
            #sum of query reshaped and value reshaped (broadcasts dimensions 1 and 2 (0-indexed))
            #size of tensor c is [batchsize, 30, 30, 300]
            sum_ = tf.add(query_reshaped,value_reshaped)
```
Next, we create the rectified version of the tensor ‘sum’. We do this by taking the hyperbolic tangent (tanh) of the sum.
```
            #rectified version of sum_
            #same shape as sum_
            rectified = tf.math.tanh(sum_)
```
We pass the rectified sum through a dense layer to get scores for each word pairing.                      
```
            #calculate scores for each word pairing
            #shape of scores is [batchsize, 30, 30, 1]
            scores = keras.layers.Dense(1,activation='linear')(rectified)
```
Now, each query word (e.g. each word in the document text) is given a weighting over the value words (e.g. each word in the query text). We do this by taking the softmax of the scores tensor along axis 2.           
```
            #each query word (each word in the document text) is given a weighting over the value words (each word in the query text)
            #weights has same shape as scores [batchsize, 30, 30, 1]
            weights = tf.nn.softmax(scores,axis=2)
```
Now that we have the weights, we multiply the weights and the values to get the weighted values. Note that at this point our values (e.g. query text representation) has dimensions [batchsize, 1, 30, 300], our weights has dimension [batchsize, 30, 30, 1], and our weighted values has the shape [batchsize, 30, 30, 300]. 
```
          #multiply the weights and the values
            #value reshaped size is [batchsize, 1, 30, 300]
            #weights has same shape as scores [batchsize, 30, 30, 1]
            #weighted values has shape [batchsize, 30, 30, 300]
            weighted_values = keras.layers.Multiply()([weights, value_reshaped])
```
Now we can reduce the dimension of the weighted values along axis 2 by using the tensorflow reduce_sum function. This will return an output of the shape [batchsize, 30, 300]. 
```
          #output has shape of [batchsize, 30, 300]
            output = tf.reduce_sum(weighted_values,axis=2)

            return output
```
The following method simply selects the weight shapes, and returns the batchsize, length of words (hard coded to 30 for simplicity) and the length of the embeddings.  
```
    def select_weight_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0],30, 300) #(batch size, length of words (hard coded for simplification), length of embeddings)
```
The rest of the model is much more straight forward: first, we specify the input dimensions.  
```    #Inputs are left and right text
        input_left = keras.Input(name='text_left', shape=(300,30))
        input_right= keras.Input(name='text_right', shape=(300,30))
```
We only weight the right inputs (document text). We use the Keras lambda layer to pass our select weight method, the weight shape, and the inputs.  
```
      #Attention layer
        Weighted_right = keras.layers.Lambda(select_weight,
                    output_shape=select_weight_shape)([input_left, input_right])
```
Now we want to flatten our matrices before we feed them to a dense layer for dimensionality reduction.  
```
'''Flatten matrices before feeding to dense layer'''
        flatten_layer = keras.layers.Flatten()
        flat_left = flatten_layer(input_left)
        flat_right = flatten_layer(Weighted_right)
       
        '''Pass through a dense layer for dimensionality reduction'''
        dense = keras.layers.Dense(
        64, activation='elu', use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None
        )
       
        dense_left_result = dense(flat_left)

        dense_right_result = dense(flat_right)

```
Then we pass the results of the dense layer to a cosine similarity function to determine how similar the texts are to one another.
```
'''Calculate cosine similarity'''
        dotted = keras.layers.Dot(axes=1, normalize=True)([dense_left_result, dense_right_result])
       
        self._backend = keras.Model(inputs=[input_left,input_right], outputs=dotted, name="attention_model")
```

Now that our model is defined, we’ll need to operationalize it with user code. 
First, we import pandas, and read in a csv containing the labels as outline in previous sections.
```
import pandas as pd

preprocessor = AttentionModelPreprocessor(30)

df = pd.read_csv(r"international-goldstandard.csv")
```
We divide our data into three segments: training (80% of the data), validation (10% of the data), and testing (10% of the data).
```
train_size = .8
valid_size = .1
test_size = .1
train_offset = int(len(df)*train_size)
valid_offset = int(len(df)*(valid_size) + train_offset)
test_offset = int(len(df)*(test_size))
train_df = pd.DataFrame(df.iloc[0:train_offset])
valid_df = pd.DataFrame(df.iloc[train_offset: valid_offset])
test_df = pd.DataFrame(df.iloc[valid_offset:])
valid_df.reset_index(inplace = True, drop=True)
test_df.reset_index(inplace =True, drop=True)
```
Then we convert our data as MatchZoo data pack objects
```
train_pack = mz.pack(train_df)
valid_pack = mz.pack(valid_df)
test_pack = mz.pack(test_df)
```
Using our preprocessor defined in the code above, we preprocess the datapack objects.
```
train_pack_processed = preprocessor.transform(train_pack, verbose=0)
valid_pack_processed = preprocessor.transform(valid_pack, verbose=0)
test_pack_processed = preprocessor.transform(test_pack, verbose=0)
```
Next, we need to define what our task is. In this case, we use ranking tasks, and define our metrics. Mean Average Precision, Mean Reciprocal Rank, and Normalized Discounted Cumulative Gain @ 1, 3, and 5, are common metrics used in Information Retrieval. They all point to how well the model returns relevant matches.
```
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
ranking_task.metrics = [
        mz.metrics.MeanAveragePrecision(),
        mz.metrics.MeanReciprocalRank(),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5)
    ]
```
Initialize the train generator (learn more about the MatchZoo train generator here: URL) Here, we tell MatchZoo that it will be training on the train_pack_processed data pack, and passing in the batch_size.
```
train_generator = mz.DataGenerator(
            train_pack_processed,
            mode='pair',
            num_dup=5,
            num_neg=1,
            batch_size=10
        )
```
Now we’re reading to create an instance of our class, Attention Model, defined in previous code, set the model’s parameters, build, and compile it. 
```
model = AttentionTextMatchingModel()
model.params['task'] = ranking_task
model.build()
model.compile()
```
Next, we’ll need to ‘unpack’ our validation data in order to pass it to the evaluate function. The evaluate method will be passed to the model’s fit generator, and will be used as callbacks during training.
```
valid_x, valid_y = valid_pack_processed.unpack()

evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=len(valid_y))
```
Now it’s time to train our model. We pass the fit generator our train generator, indicate the number of epochs, the number of workers, whether to use multiprocessing, and whether to shuffle the data during epochs. 
```
history = model.fit_generator(train_generator, epochs=10, callbacks=[evaluate], workers=4, use_multiprocessing=True, shuffle=True)
```
Now we unpack our test data pack, and use the model.evaluate() method to get our MAP, MRR, and NDCG @1, 3, and 5 metrics on our test data: 
```
test_x, test_y = test_pack_processed.unpack()

performances=model.evaluate(test_x, test_y)
```
Our original data should still be in memory in the test_df data frame (from the first step in the user code.) We can save the output of our model (e.g. the cosine similarity between each query and document texts) to a column in the data frame named predictions to further evaluate our model. 
```
test_df['predictions'] = model.predict(test_x)
```
Now, we’re ready to save our model’s results to a CSV: 
```
test_df.to_csv(‘AttentionModelResults.csv'
```

