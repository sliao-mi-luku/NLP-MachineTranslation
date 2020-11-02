# NLP-MachineTranslation

**Repository Updated on 10/31/2020**

## References

**1. Udacity**

This project (and the dataset) comes from one of the projects in Udacity's NLP Nanodegree course I took in Septenmber, 2020.

**2. Transformer tutorial**

I learned the transformer implementation from the tutorial notebook below:

https://www.tensorflow.org/tutorials/text/transformer

**3. BERT**

I'm learning how to build a BERT model:

https://www.tensorflow.org/official_models/fine_tuning_bert


## The Project

The goal of this project is to use deep neural network architectures to translate English sentences into French.

## The Dataset

This project uses the dataset provided by [Udacity's NLP Nanodegree Course](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892). The dataset is a subset extracted from the [WMT](http://www.statmt.org/) dataset. It contains 137,861 English sentences and their translations in French.

Most of the notebook was run on Google Colaboratory. To run the notebooks in this repo, simply upload the training data (`small_vocab_en`, `small_vocab_fr`) onto the Google Colab workspace (assuming that you have those files). If you don't have the files I used, you can upload your own dataset (even in different languages), and rewrite your data preprocessing codes.

## Data Preprocessing

### Data split
I splitted the datasets (137,861 sentence pairs) into:

- training dataset of 120,000 sentences,
- validation dataset of 7,861 sentences, and
- test dataset of 10,000 sentences

### Tokenization
I used [SubwordTextEncoder](https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder) to tokenize the English and French sentences in the training data.

The tokenizer tokenizes the word into tokens between 1 (inclusuce) and `vocab_size` (exclusive). If the tokenizer encounters a word (from the validation or test data) it hasn't seen, it'll tokenize the word by its subwords.

**Update** This tokenizer method has been deprecated, which can still be called by importing from `tfds.deprecated.text.SubwordTextEncoder`

``` python3
## Subword Tokenizer
english_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((eng.numpy() for eng, fr in train_dataset), target_vocab_size = 2**13)
french_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((fr.numpy() for eng, fr in train_dataset), target_vocab_size = 2**13)

## Vocabulary size (+2 for start and end)
input_vocab_size = english_tokenizer.vocab_size + 2
output_vocab_size = french_tokenizer.vocab_size + 2
```

## Models

This project uses different models to tackle the problems:

| File Name | Model Description |
| ----------- | ----------- |
| EncoderDecoder_English_French_translator_colab.ipynb | Deep RNN (GRUs or LSTMs) |
| Transformer_English_French_translator_colab.ipynb.ipynb | [Transformer (Google)](https://arxiv.org/abs/1706.03762) |

#### Encoder-Decoder Architecture (with bidirectional GRU or LSTM layers)

``` python3
class EncoderDecoder(tf.keras.layers.Layer):

    def __init__(self, rnn_class, output_dim, embedding_dim, encoder_units, decoder_units, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        self.Embedding = Embedding(input_vocab_size, embedding_dim)

        if rnn_class.upper() == "GRU":
            self.Bidirectional_encoder = Bidirectional(GRU(encoder_units))
            self.Bidirectional_decoder = Bidirectional(GRU(decoder_units, return_sequences = True))
        elif rnn_class.upper() == "LSTM":
            self.Bidirectional_encoder = Bidirectional(LSTM(encoder_units))
            self.Bidirectional_decoder = Bidirectional(LSTM(decoder_units, return_sequences = True))
        
        self.RepeatVector = RepeatVector(output_dim)
        self.TimeDistributed = TimeDistributed(Dense(output_vocab_size))
        self.Activation = Activation('softmax')

    def call(self, x):
        x = self.Embedding(x)
        x = self.Bidirectional_encoder(x)
        x = self.RepeatVector(x)
        x = self.Bidirectional_decoder(x)
        x = self.TimeDistributed(x)
        y = self.Activation(x)
        return y

```

#### Transformer

#### BERT


## Model Performance and Evaluation

| Metric | EncoderDecoder (biGRU) | EncoderDecoder (biLSTM) | Transformer | 
| ----------- | ----------- | ----------- | ----------- |
| Epochs | 20 | 20 | ----------- | ----------- |
| Training Acc | 98.7% | 99.1% | ----------- |
| Validation Acc | 93.2% | 98.0% | ----------- |
| Test Acc | 98.1% | 99.9% | ----------- |
| Test BLEU| 0.955 | 0.962 | ----------- |


## Conclusion
