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



```import pandas as pd```
