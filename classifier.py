import sys
import string
import math

'''
Tuning of the Classifier:

Original validation stats (k = 1, c = 1):
Precision:0.950819672131 Recall:0.865671641791 F-Score:0.90625 Accuracy:0.978456014363

Tuning is done on the dev data. Started by increasing k by a lot and comparing stats. First 
choice was 10 which saw a minor decrease in stats. Second choice was 5 which was better than 
10 but not quite as good as k=1. k=3 produced the same as k=5 while k=2 produced the same as
k=1. This implies that for a c=1, k=1 is best.

Next I tried adjusting c, first setting it to 10. This had a dramatic decrease in some statistics 
though precision was 1. I tried adjusting k with c=10 to see the affect and it produced the best 
results when 7<k<12. This proved that k could be benificial depending on c but the performance still
wasn't as good as with k=1, c=1.

Since k=1 is the minimum for k, I decided to next try decreasing c. I started with a small change 
this time and set c to 0.1. Playing around with k, I found that the best k for c=0.1 is 1. This 
increased performance, so I decreased to 0.01 which saw a slight decrease in performance from 0.1
while the best k was still 1. To make sure it wasn't better to go lower for c, I tried 0.001. This 
also had a slight decrease in performance and had an optimal k of 1. I  decied to try values around 
0.1 for c. For around 0.1 (0.08, 0.09, 0.11), I was able to achieve the same performance metrics as
with c=0.1. I was also able to achieve these with k=1 or, in some cases, k=2 or k=3.

Because of these results, I am setting the optimal values for c and k to:
k=2, c=0.1

On the validation set, this gives: 
Precision:0.967213114754 Recall:0.880597014925 F-Score:0.921875 Accuracy:0.982046678636

On the test set, this gives:
Precision:0.964912280702 Recall:0.873015873016 F-Score:0.916666666667 Accuracy:0.982046678636

'''

'''
Stop Words:

For my implementation, stop words can be added by including the file name for them
as a command line argument after the test or validation set.

When adding the provided stop words and keeping k=2 and c=0.1, the dev set for validation showed a slight decrease
in performance and had the following metrics:
Precision:0.951612903226 Recall:0.880597014925 F-Score:0.914728682171 Accuracy:0.980251346499

The test set however proved better with the stop words with the following:
Precision:0.965517241379 Recall:0.888888888889 F-Score:0.925619834711 Accuracy:0.983842010772
'''

def extract_words(text):

    # Remove punctuation
    text = text.translate(None, string.punctuation)

    # Make the text lowercase
    text = text.lower()

    # Remove any trailing whitespace
    text = text.strip()

    # Split into a list of strings
    text_list = text.split()

    return text_list


class NbClassifier(object):

    def __init__(self, training_filename, stopword_file = None):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}
        self.stop_words = set()

        if stopword_file != None:
            self.load_stop_words(stopword_file)
        self.collect_attribute_types(training_filename, 2)
        self.train(training_filename)          

    def load_stop_words(self, stopword_file):

        # Open the stopword file
        with open(stopword_file, 'r+') as file:

            # Read in the lines
            for stop_word in file:

                # Do a bit of preprocessing 
                stop_word = stop_word.strip()
                stop_word = stop_word.lower()
                stop_word = stop_word.translate(None, string.punctuation)

                # Add the stop word to the set
                self.stop_words.add(stop_word)

    def collect_attribute_types(self, training_filename, k):

        # Data structures to count the frequency of words
        vocabulary = set()
        frequencies = {}

        # Open the training file and read in the lines
        with open(training_filename, 'r+') as file:

            # Read in the lines
            for lines in file:

                # Get the words
                words = extract_words(lines)

                # Go through the words (ignoring the spam/ham classifier)
                for each_word in words[1:]:

                    # If the word is a stop word, go right to the next word
                    if each_word in self.stop_words:
                        continue

                    # Update the frequency of the word
                    if each_word in frequencies:
                        frequencies[each_word] += 1
                    else:
                        frequencies[each_word] = 1

                    # Add the word to the vocab set if the frequency matches k
                    # - Any more and its already been added
                    if frequencies[each_word] == k:
                        vocabulary.add(each_word)

        # Set the attribute types variable
        self.attribute_types = vocabulary

    def train(self, training_filename):

        # Laplacian Smoothing Constant
        c = 0.1

        # Local variables for working with data
        count_word_given_label = {}
        count_word_label = {}
        count_label_prior = {}
        count_messages = 0

        # Add all the words from the attributes to the dictionary (set counts to 0)
        for word in self.attribute_types:
            count_word_given_label[(word, 'spam')] = 0
            count_word_given_label[(word, 'ham')] = 0
            count_word_label['spam'] = 0
            count_word_label['ham'] = 0
            count_label_prior['spam'] = 0
            count_label_prior['ham'] = 0

        # Open the training file and read in the lines
        with open(training_filename, 'r+') as file:

            # Read in the lines
            for lines in file:

                # Get the words
                words = extract_words(lines)

                # Iterate through the words in the line
                for each_word in words[1:]:

                    # Update the count_word_label dictionary
                    if words[0] in count_word_label:
                        count_word_label[words[0]] += 1

                    # Update the count_word_given_label dictionary
                    if (each_word, words[0]) in count_word_given_label:
                        count_word_given_label[(each_word, words[0])] += 1

                # Update the count_label_prior dictionary
                if words[0] in count_label_prior:
                    count_label_prior[words[0]] += 1
                count_messages += 1

        # Go through the words and calculate probabilities
        for key, values in count_word_given_label.iteritems():

            # Get the probability
            p = float(values + c) / (count_word_label[key[1]] + c*len(self.attribute_types))

            # Add the probability to the dictionary
            self.word_given_label[key] = p

        # Go through the labels and calculate their values
        for key, values in count_label_prior.iteritems():

            # Get the probability
            p = float(values) / count_messages

            # Add the probability to the dictionary
            self.label_prior[key] = p

    def predict(self, text):

        # Classification dictionary
        classification_dict = {}

        # Get the words from the text
        words = extract_words(text)

        # Compute the log of the joint probability
        for label in self.label_prior:

            # Get the log of the probability of the label
            running_sum = math.log(self.label_prior[label])

            # Go through the words and add up the probability of the word at that label
            for each_word in words:

                # Don't try to find the word if its not in the dictionary
                if (each_word, label) not in self.word_given_label:
                    continue

                # Add in the log of the probability
                running_sum += math.log(self.word_given_label[(each_word, label)])

            # Once the logs of probability have been added up, add that label to the dictionary
            classification_dict[label] = running_sum

        # Return the classification dictionary
        return classification_dict

    def evaluate(self, test_filename):

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        # Open the validation file and read in the lines
        with open(test_filename, 'r+') as file:

            # Read in the lines
            for lines in file:

                # Split the line into message text and label
                label, text = lines.split('\t', 1)

                # Get the prediction
                classification = self.predict(text)
                
                max_val = -10000
                max_key = ''

                # Get the max classifier
                for key, vals in classification.iteritems():
                    if vals > max_val:
                        max_val = vals
                        max_key = key

                # Compare the key (label) to the label and update stats

                # True Positive
                if max_key == 'spam' and max_key == label:
                    tp +=1

                # False Positive
                elif max_key == 'spam' and max_key != label:
                    fp += 1

                # True Negative
                elif max_key == 'ham' and max_key == label:
                    tn += 1

                # False Negative
                elif max_key == 'ham' and max_key != label:
                    fn += 1

        # Calculate Stats
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        fscore = 2 * float(precision) * recall / (precision + recall)
        accuracy = float(tp + tn) / (tp + tn + fp + fn)
        
        return precision, recall, fscore, accuracy


def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":
    
    if 3 < len(sys.argv):
        classifier = NbClassifier(sys.argv[1], sys.argv[3])
    else:
        classifier = NbClassifier(sys.argv[1])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)
