Naive Bayes Classifier

Given a set of training documents, a gzip file of essays, creates a trained model using a set vocab list and bag of words. Then classifies a given essay using the trained model to predict the year an essay came form. 

Classifier uses MLE function of log probabilities to predict year: argmax Sumation(log P(x|y)) + log(y)
