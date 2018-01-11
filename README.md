# COMS-W4701-Bayesian-Classification-of-Spam-Text-Messages
COMS W4701 Artificial Intelligence Programming HW 3: Bayesian Classification of Spam Text Messages

### Objective
Using a Beyesian Classifier, determine whether a text message is spam or not (labeled as "ham" in the datasets). The dataset is split into three parts. A training set to train our classifier on, a validation set to adjust parameters, and a test set to test the the classifier. The stop words file was provided by our professor but any stop words files in the same format can be used.

### Outcome
See the commented sections at the top of the python file for degree of success. Overall, the classifier worked very well.

### How To Run
To train the model and evaluate it on the validation set, use:
```
python classifier.py train.txt  dev.txt
```
Once the model's parameters are fine tuned using the validation set, use the following to train and evaluate on the test set:
```
python classifier.py train.txt  test.txt
```

### Dataset
The included dataset can be found here:

https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

It is also fully examined in this work:

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. "Contributions to the Study of SMS Spam Filtering: New Collection and Results".  Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11).
