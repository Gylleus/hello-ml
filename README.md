# Hello ML

This repository contains a simple implementation of a custom Tensorflow estimator. The estimator is a DNN with a flexible amount of layers and their respective dimensions.

The estimator tries to predict the dollar amount of a purchase made on a certain Black Friday based on information about the customer. The data used is public and was downloaded from [Kaggle](https://www.kaggle.com/mehdidag/black-friday?).

This model was developed and tested using Python 3.6.7.

### Create and train model

```bash
python3 dnn_estimator.py
```