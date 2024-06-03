# Embeddings Transformations for Sentiment Lexicon Enrichment

TLDR: 

**There exists a vector v, such that translating a negative word n by v leads to the vicinity of the antonym of n.**

Examples:

* bad + _v_ ≈ good
* boring + _v_ ≈ interesting
* stupid + _v_ ≈ smart

## Introduction

Sentiment Analysis (aka Polarity Detection) is the task of deciding whether a given text is positive, negative or neutral. I was recently tasked with building a system to perform this analysis for a specific domain.

We set ourselves a goal of doing this without annotated examples — i.e without using supervised learning at first. I looked into classical approaches that focused on domain knowledge, and how unsupervised learning (word vectors), could be used to automatically grow our sentiment lexicon.

## Sentiment Lexicon

Sentiment Lexicons are a simple beast. They consist of a mapping from a word to its polarity. The polarity score could be categorical (positive, neutral, negative) or numerical (e.g. on a scale from -5 to 5).

[![Sentiment Lexicon](/images/catnum.webp)](/images/catnum.webp) 


Sentiment analysis is then some aggregation on the scores of the words in the text.
Lexicons can be created both manually by domain experts or algorithmically via various statistical measures. 
In the rest of the article we will focus on _enriching an existing sentiment lexicon_ via transformations in word embedding space.

## Word Embeddings

Word embeddings are representations of words in a high dimensional space. Each word is associated with a vector and semantically related words are close in embeddings space.

Word embeddings have been around for a while, but it was a 2013 paper 
[“Efficient Estimation of Word Representations in Vector Space”](https://arxiv.org/abs/1301.3781) which brought them to the spotlight. Embeddings are now a standard part of most deep learning models dealing with NLP.

Word vectors can be derived via various algorithms. 
Most of them rely on the **distributional hypothesis** which states that words that are used and occur in the same contexts tend to purport similar meanings. 
The most popular embeddings algorithms are:

* Continuous Bag of Words
* Skipgram Model
* GloVe
* FastText

The cool thing about word embeddings is that they encode semantics 
and it is even possible to carry out arithmetical operations 
which preserve the semantic structure.
The most famous example is that _“king is to queen as man is to woman”_:

**king − queen ≈ man − woman**

We shall leverage this property to enrich our sentiment lexicon.

## Lexicon Enrichment

Lexicon enrichment will be achieved via two operations:

* Search for synonyms by looking at the most similar vectors to known positive or negative words

[![nn](/images/nn.webp)](/images/nn.webp)
Nearest Neighbors of the word "great" in embeddings space

* Search for antonyms by looking at the most similar vectors after translating by the neg2pos vector v or the pos2neg vector **-v**

[![antonyms](/images/antonyms.webp)](/images/antonyms.webp)
Nearest neighbours of the word “great” after translating by the pos2neg vector -v


The steps needed to achieve the lexicon enrichment are:

1. Load the pretrained word embeddings. We will be using the top 100K words from the pretrained Glove Embeddings (glove.42B.300d.zip)
```python
from gensim.models import KeyedVectors
pretrained_embeddings = KeyedVectors.load_word2vec_format('glove.100k.300d.txt')
```

2. Find the vector v by taking the mean of a small set of predefined antonym pairs
```python
import numpy as np
sentiment_pairs = [('good', 'bad'), ('awesome', 'awful'), ('interesting', 'boring'), ('happy', 'sad'), ('beautiful', 'ugly')]
v = np.mean([  pretrained_embeddings[x[0]] - pretrained_embeddings[x[1]] for x in sentiment_pairs], axis=0)
```
3. Define the neg2pos and pos2neg functions as simple translations by the vector v.

As it turns out, translating by the neg2pos vector leads to a more positive context, but it is still in the vicinity of the original word, and thus to its closest words/synonyms. I’ve proposed a simple way to filter them out — if a word in the new positive context is also present in the original negative context, but the score has decreased, then presumably it is a synonym of the negative word, and should be ignored when searching for antonyms.

```python
import pandas as pd


def pad2len(arr, desired_len):
    return arr + [""]* (desired_len - len(arr))

def translate(w, vector):
    closest = pretrained_embeddings.similar_by_word(w)
    closest_dict = dict(closest)
    translated = pretrained_embeddings.similar_by_vector(pretrained_embeddings[w]+vector)
    translated = [x for x in translated if x[0] != w]
    filtered = [x for x in translated if (x[0] not in closest_dict or closest_dict[x[0]] < x[1])]
    translated = pad2len(translated, len(closest))
    filtered = pad2len(filtered, len(closest))
    return pd.DataFrame({'closest': closest, 'translated': translated, 'filtered':filtered}, columns=['closest', 'translated', 'filtered'])

def neg2pos(w):
    return translate(w, v)

def pos2neg(w):
    return translate(w, -v)
```

The tables below show how the transformations work for some examples both in the positive and 
negative directions. The `start_word` column lists the word from which the transformation 
was initiated. The `closest` column shows the words closest to the start_word in embeddings space. 
In the common case, they should be the synonyms of the start word. The `translated` column shows the words closest to the (start_word+v) point in embeddings space. 
The theory is that they should represent the antonyms of the start word. 
Unfortunately, they still contain some of the synonyms of the start_word. The last `filtered` column shows the results with the synonyms filtered out via the technique proposed above.

<script src="https://gist.github.com/mboyanov/89c07f69d8fea47c2b5c1d6c6167ec3b.js"></script>
<script src="https://gist.github.com/mboyanov/60dc5514d5ec72cff9c03fdead5d81d6.js"></script>

## Caveats
My initial experiments seem to work best with adjectives. It’s possible that the antonym vector for nouns is different.

## Future Work
In the near future I plan to examine some nice decompositions of the polarity bearing word vectors and to introduce a density based method to discover and score polarity for words in the embeddings vocabulary.