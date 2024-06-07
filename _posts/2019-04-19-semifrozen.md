# Semi-Frozen Embeddings for NLP Transfer Learning

## TL;DR
_Semi-frozen embeddings_ split the pretrained embeddings into two groups: 
a _trainable group_ and a _frozen group_ so that words missing in the original pretrained vocabulary can be trained, while the rest stay frozen.

Code available [here](https://github.com/mboyanov/semifrozen-embeddings/blob/master/Task%202-SemiFrozen%20Embeddings.ipynb).

[![Embeddings](/images/semifrozen.webp)](/images/semifrozen.webp)

## Introduction & Problem Statement

The idea behind _“transfer learning”_ is to take a state of the art model pre-trained on a massive dataset and then fine-tuning the model on a new related task where the training data is insufficient to train an analogous model. Transfer learning is gaining more and more traction in the NLP world with models such as [BERT](https://arxiv.org/abs/1810.04805), [Elmo](https://arxiv.org/abs/1802.05365) and [ULMFit](https://arxiv.org/abs/1801.06146) showing the enormous potential.

To be fair, transfer learning has been around for quite some time but in a more relaxed setting — the usage of pretrained word embeddings.

Unfortunately, there is a huge problem — particularly when transferring knowledge from one domain to another — the problem of _vocabulary mismatch_. Very often, the target domain contains words which are quite specific and important, but they are missing from the original domain. It is vital to incorporate these words in the model, but their absence from the pretrained embeddings vocabulary makes it problematic.

There are three obvious options for modeling them:

1. Discard the pretrained embeddings, start from scratch with a random init.
2. Set the unknown words to the mean of the pretrained embeddings and keep them frozen
3. Set the unknown words to the mean of the pretrained embeddings and unfreeze the entire layer

All of these have their drawbacks:

Option 1. _discards_ any notion of transfer learning.    
Option 2. treats all unknown words the same and keeps them frozen, thus _failing to model_ them effectively.    
In my experiments, Option 3. would eventually lose all prior knowledge and _overfit_ on the training set.    


Thus, I propose a compromise: **_Semi-Frozen Embeddings_**. 
The semi-frozen embeddings module would split the embeddings into two groups: a frozen group and a trainable group.
The _frozen group_ will consist of the pretrained word embeddings and will be (duh) frozen — _it will NOT be updated during training_. 
The _trainable group_ will contain the new words that need to be finetuned. They are initialized with the mean of the pretrained embeddings and are _updated during training_.

I implemented such a module in pytorch and evaluated the four methods on the [Hack the News datathon task 2](https://www.datasciencesociety.net/hack-news-datathon-case-propaganda-detection/), organized by the [Data Science Society](https://www.datasciencesociety.net/) . A simple bi-LSTM model leveraging the semifrozen embeddings gets us to 3rd place in the leaderboard(team Antiganda)!

The model was trained via the awesome [fast.ai](https://github.com/fastai/fastai) library and the [glove](https://nlp.stanford.edu/projects/glove/) word vectors were used for pretrained embeddings. Please check [the notebook](https://github.com/mboyanov/semifrozen-embeddings/blob/master/Task%202-SemiFrozen%20Embeddings.ipynb) for additional details.


## Experiments & Results
The four models were run for a total of 16 epochs with a batch size of 64 and a max. learning rate of 3e-3.

What is obvious about the model is that the Random init and Unfrozen models quickly overfit on the training set as the validation loss diverges.

[![Results](/images/semifrozen-random-init.webp)](/images/semifrozen-random-init.webp)

On the contrary, both the Frozen and Semi-frozen models achieve much lower validation loss with the Semi-Frozen model achieving slighly better results.

[![Results](/images/semifrozen-results.webp)](/images/semifrozen-results.webp)

## Conclusion
Semi-frozen embeddings can be used to finetune pretrained word embedding to a new domain, leaving the original word embeddings unchanged.

This approach is more stable than unfreezing the entire embeddings layer and helps avoid overfitting.