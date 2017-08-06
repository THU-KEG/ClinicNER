# ClinicNER
ClinicNER experiments

## I2B2K 2010 Data set
> First, run i2b2k_data_preprocess.py to get BIO format data from i2b2k data set.

> For Evaluation: perl conlleval.pl -d '\t' < i2b2k/2010/result.txt

> This file "result.txt" is in BIO format

## Denpendency
>NLTK,numpy,scipy,python-crfsuite
>You can use `pip` to install these packages.

## Quick Start
>git clone
>python nertest.py

## Some results
- accuracy:  91.11%; 
- ALLprecision:  81.64%; recall:  76.20%; FB1:  78.83
- PROBLEM: precision:  79.55%; recall:  77.61%; FB1:  78.57  12285
- TEST: precision:  83.12%; recall:  74.46%; FB1:  78.55  8264
- TREATMENT: precision:  83.20%; recall:  76.03%; FB1:  79.45  8538


## Important Code files

- i2b2k_data_preprocess.py : raw data format to BIO format, and generate vocab.txt

- data_utils.py : the util to read vocab, embedding, category data, compute category center.

- nertest.py : the main file of this experiment, includes feature generation and training.
