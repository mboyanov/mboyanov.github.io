## Let's build Bg2Vec: Preparing the Training Data 

Hello and welcome to the second post in the Bg2Vec series. In this post, we will prepare the training data for the Bg2Vec model.

As a reminder, the plan is as follows:

1. Part 1 - Overview & Objectives. 
2. Part 2 - Preparing the training data (this post)
3. Part 3 - Masked Next Token Prediction training
4. Part 4 - SimCSE training
5. Part 5 - Evaluation of the resulting text encoder


Similar to the original llm2vec paper, we will use the [Bulgarian Wikipedia](https://bg.wikipedia.org/) as our finetuning data.

First, we will download the latest dump of the Bulgarian Wikipedia and push it to huggingface.co. (Check out the result [here](https://huggingface.co/datasets/mboyanov/bgwiki).)

Afterwards, we will preprocess the dataset so it is suitable for training the model with llm2vec and bggpt. 
To get an overview of the preprocessing steps, please consider the following figure:

![Preprocessing](/images/preprocessing.png)

## Preparing the Data

To get our hands on the data, we can go to [dumps.wikimedia.org](https://dumps.wikimedia.org) and download the [latest dump](https://dumps.wikimedia.org/bgwiki/20240501/) of the Bulgarian Wikipedia. 
The dump is in XML format and contains all the articles in the Wikipedia.

We will use the `wikiextractor` tool to extract the text from the XML dump. The tool can be found [here]()

Then we can use it as follows:

```
wikiextractor bgwiki-20240420-pages-meta-current.xml  --json -b 5000M
```

This will extract the text from the XML dump and save it in JSON format in a directory called `text`. 
The text will be split into files of 5000MB each which results in a single file in our case. 

Inspecting the data, we can see that we have quite a bunch of empty articles (which simply redirect to other articles).
We will remove these articles as they do not contain any useful information.

Finally, we will save the resulting dataset in .parquet file so that it works better with the huggingface datasets library.

For reusability I have already prepared the dataset and uploaded it to the huggingface datasets library. You can find it [here](https://huggingface.co/datasets/mboyanov/bgwiki).

And to use it in your code, you can simply issue:

```python
from datasets import load_dataset

dataset = load_dataset("mboyanov/bgwiki")
```


## Preprocessing

Preprocessing the data consists in the following steps:

1. Tokenization - tokenization is the process of splitting the text into tokens. Tokenization is model-specific, so we will use the tokenizer from the BgGPT model.
2. Chunking - we will group the articles into chunks of 512 tokens. This is an optimization step related to optimal usage of the GPU memory.

Let's start with the tokenization by loading the BgGPT tokenizer.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("INSAIT-Institute/BgGPT-7B-Instruct-v0.2")
```

Let's explore how we can use the tokenizer. The tokenizer has a `tokenize` method which splits the text into tokens 
and a `__call__` method which returns the input_ids and attention_mask of the tokenized text. 
Note how we can also specify `return_tensors='pt'`, so the return type is pytorch tensors.
There is also a `decode` method which converts the input_ids back to text.

```python
sent = "Здравейте, как сте?"
tokenizer.tokenize(sent)
> ['▁Здраве', 'йте', ',', '▁как', '▁сте', '?']

tokenizer(sent)
> {'input_ids': [1, 35834, 32111, 28725, 8600, 19776, 28804], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}

tokenizer(sent, return_tensors='pt')
> {'input_ids': tensor([[    1, 35834, 32111, 28725,  8600, 19776, 28804]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}

tokenizer.decode([1, 35834, 32111, 28725, 8600, 19776, 28804])
'<s> Здравейте, как сте?'
```

Ok, so now we need to apply the tokenization to the dataset. We will use the `map` method of the dataset object to apply the tokenization to all articles.

```python
def tokenize_dataset(raw_datasets, tokenizer, data_args, training_args, text_column_name="text"):
    column_names = list(raw_datasets["train"].features)
    assert text_column_name in column_names, f"Provided text_column_name {text_column_name} not in dataset"
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name], return_special_tokens_mask=True
        )

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        
    return tokenized_datasets
```

Now that we have tokenized the dataset, we can proceed with the chunking. We will group the articles into chunks of 512 tokens.

```python
from itertools import chain
def group_texts(examples:LazyBatch, max_seq_length=1024):
    """
    This function will receive a batch of texts and return a list of chunks of texts that have length max_seq_length.
    Intended usage with `datasets.map(ds, batched=True)`

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map
    """
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [
            t[i : i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
        for k, t in concatenated_examples.items()
    }
    return result

```

After applying the chunking, we can save the resulting dataset by using the `save_to_disk` method of the dataset object.

Stay tuned for the next part in the series, where we will fine-tune the BgGPT model using the Masked Next Token Prediction objective.

## Useful links

1. [BgGPT](https://bggpt.ai/)
2. [Huggingface datasets library](https://huggingface.co/docs/datasets/)
3. [Huggingface transformers library](https://huggingface.co/transformers/)
4. [bg2vec repository](https://github.com/mboyanov/bg2vec)
5. [Bulgarian Wikipedia dataset](https://huggingface.co/datasets/mboyanov/bgwiki)