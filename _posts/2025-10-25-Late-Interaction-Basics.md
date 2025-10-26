# Late Interaction Basics

I am fascinated with embeddings. Do you remember those word analogy examples? Paris is to France as Rome is to Italy? 
Fascinating. Embeddings have been present in one form or another in most of my work, blog posts and talks.
Their ability to capture semantic relationships makes them a versatile tool that can be applied in many 
ways.

I also love full text search. I remember long nights in the lab fiddling with tokenizers and inverted
indices in an attempt to build a simple full text search engine. I think that's what inspired me to
pursue NLP.

We're here today to talk about a technique which brings these two together: Late Interaction. Classical search systems 
may fail to understand the query or the documents in some major way - they have limited understanding of the context or the meaning of the document. 
On the other hand, vector search loses context in the aggregation/pooling step.

What if there was a middle ground? Late interaction search is an emergent technique which
allows us to mitigate these problems.

This post will introduce the main ideas behind late interaction search and how it can
improve on classical full text search and classical vector search.

Papers
The main papers we will focus on are:
1. ColBERT: Eicient and Eective Passage Search via Contextualized Late Interaction over BERT (Stanford, 2020)
2. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction (Stanford & Georgia Institute of Technology, 2021)

## Comparison
Please study the following graphic in detail. It shows the differences between full text search, vector search 
and late interaction search.
![Comparison](images/LateInteractionComparison.png)

## Late Interaction mechanism


## Speed and quality

## Size

## Thinking tokens / Query Augmentation

In the pre-reasoning models era, there was often a recommended technique to ask the model to explain his reasoning or solution. E.g. add “Let’s think step by step”. This allows the model to use more compute to come up with an answer - see Deep Dive into LLMs like ChatGPT
 by Andrej Karpathy.

Here we have a similar approach: we append a few <pad> tokens to the input and use them to query the vector store as well.

Example Query: 

neoshare macht spass 
=> [neoshare,macht,spass, <pad>, <pad>, <pad>] 
=> query embeddings tensor 6 x emb_dim
=> 6 vector store lookups
=> maxsim

## Further reading

Further study
ColPali: Efficient Document Retrieval with Vision Language Models - they do it directly in the vision space

DeepSeek OCR?