# Forward + Backward Character Language Models for Conjoined Word Separation with fast.ai

Conjoined words are _words that wrongly consist of two words_. Examples are “theproblem”, “extremecircumstances”, “helloworld”, etc. This article explores how we can use a pair of _character language models_ (one forward and one backward) to detect them and separate them.

The language models are trained via the powerful new [fastai](https://github.com/fastai/fastai) v1 library
and pytorch. Please refer to the [Jupyter notebook version](https://gist.github.com/mboyanov/fedf151d1702257d8ec0856f629c3cd5) of this article for the code and implementation details.


## The Problem
At [Commetric](https://commetric.com/), we process text data gathered from news web sites to make various analyses on behalf of our clients. We have several systems processing this news feed to assign classes, perform deduplication or group articles into clusters. Thus, the quality of the data is of paramount importance. Conjoined words are a problem for NLP systems, especially those based on deep learning. The reason is that these solutions rely either on a fixed vocabulary or have some minimal number of times that a word needs to appear before they can start using it. Since conjoined words are quite rare and random, they end up being _out-of-vocabulary_ words for these systems. However, sometimes they contain the necessary information to infer the correct class (if only they weren’t conjoined and treated as separate tokens!).

The appearance of such words might be due to spelling errors, but I suspect that it is a by-product of the crawling process that gathers the data. In any case, driven by the “garbage in, garbage out” mentality, it will be beneficial for NLP systems to receive a cleaned up input and thus it is important to find and split the conjoined words.

## The Solution
I propose leveraging the predictive power of a pair of character language models to identify potential split points.

A _character language model_ is a function which accepts a string of text and then outputs the probability distribution of the next character. Essentially, it is an algorithm which “guesses” which character comes next. It’s similar to playing Hangman with only the last letter missing, e.g. in the sentence “what is the meaning of lif_” a good language model would output a high probability for the character ‘e’, a lower probability for the character ‘t’ and even lower for other symbols.

A _backwards language model_ is the same as a language model, but it receives the text in reverse. In effect, it’s goal would be to predict the initial letters of a given text sequence. For example, “_ay the force be with you” should result in a high probability for the character ‘m’.

To find potential split points, the two language models are combined in the following manner:

* Compute forward predictions via the forward language model
* Compute backward predictions via the backward language model
* Align the two predictions
* The potential split points are the positions where the two language models agree that the character should be a ‘ ‘

# ![Conjoined Words](/images/tesla.webp)
<div class="figcaption">The split points are the indices at which the forward and backward language models agree that a space should be inserted. In this example, only the locations marked with a rectangle will be inspected for a potential split: “pre market” and “after ceo”</div>


Now that the potential split points are identified, the split is applied only if the following conditions are met:

1. The original word without the split is not in the vocabulary
2. Both the left and right words are in the vocabulary
3. Inserting the space leads to a lower average log likelihood for the two language models


Conditions 1, 2 are more conservative and are a safety and sanity check that the transformation will indeed improve the quality of the data, whereas the third condition double-checks that inserting the space still makes sense for the two language models.

## Results


The most satisfying and frustrating aspect of machine learning is inspecting the results to try and pick up patterns where the algorithm performs fine and where it performs badly, so that it can be improved.

The following are some examples where the conjoined words were identified and split correctly:

* examiningthe => examining the
* androidcommunity => android community
* sectorstechnology => sectors technology
* afterceo => after ceo

We see that the system is quite capable of identifying conjoined words and applying the spaces.

Next follow some questionable examples:

* deathmatch => death match
* cliffhanger => cliff hanger
* newseditor => news editor
* battleships => battle ships
* spokespersons => spokes persons

Even though the transformations produce coherent splits, one can argue that the original word has a slightly different nuance. However, we should note that the original word is not in the vocabulary that the final system will use and thus using the split is definitely better than nothing.

Some undesired examples:

* energyexcel => energy excel
* onewindow => one window
* trumpcare => trump care
* cloudbolt => cloud bolt
* saskpower => sask power

The system is naturally biased towards more frequent words and thus named entities composed of several words form likely split points. As a consequence, the split might be problematic for a named entity detection system which uses the cleaned-up results.

## Future Work

The most immediate future task is to measure the effects of this preprocessing step on the final systems in terms of accuracy and f1 score.

Afterwards, the system can be improved in several ways:

* take greater care around named entities
* improve the model architecture
* perform several passes over the input, so even triple-conjoined words can be split

## Conclusion
Conjoined words are a nuisance to NLP systems, because they end up being out-of-vocabulary words.

Detecting and separating them is a hard task, but using a pair of language models is a viable technique to tackle the problem.