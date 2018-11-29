import pandas as pd
from collections import Counter
import nltk
import glob
import sys
from random import shuffle
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import operator
from nltk.corpus import stopwords
import string
import re
from nltk.corpus import wordnet

nltk.download('averaged_perceptron_tagger', quiet=True)
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

class BaseLine:
    def __init__(self, version, input_filenames):
        self.version = version
        self.input_filenames = input_filenames
        self.cluster_num = 1
        self.summary = ''
        self.summary_length = 0

    def sum_leading(self):
        if self.version == 'leading':
            filenames = glob.glob(self.input_filenames)
            shuffle(filenames)
            for filename in filenames:
                if self.summary_length >= 100:
                    break
                file = open(filename, "r")
                file_content = file.read()
                raw_sents = sent_tokenizer.tokenize(file_content)
                for sent in raw_sents:
                    if self.summary_length < 100:
                        if self.summary == '':
                            self.summary = self.summary + sent
                        else:
                            self.summary = self.summary + ' ' + sent
                        self.summary_length = self.summary_length + len(word_tokenize(sent))
                    else:
                        break
            return
        else:
            return


class SumBasic:
    def __init__(self, version, input_filenames):
        self.version = version
        self.input_filenames = input_filenames
        self.cluster_num = 1
        self.vocab_dict = {}
        self.vocab_prob_dict = {}
        self.sent_df = pd.DataFrame(columns = ['processed', 'raw'])
        self.summary = ''
        self.summary_length = 0

    def pre_process_data(self):
        # build dataset
        # print("reading cluster " + str(self.cluster_num))

        filenames = glob.glob(self.input_filenames)
        # print("find files: " + str(filenames))

        for filename in filenames:
            sent_doc_df = pd.DataFrame(columns=['processed', 'raw'])
            file = open(filename, "r")
            # print("open file: " + filename)
            file_content = file.read()

            quotes =re.findall(r'\"(.+?)\"', file_content)
            quote_sents = []
            for quote in quotes:
                quote_sents.extend(sent_tokenizer.tokenize(quote))

            # newline needs to be treated as start of another sentence
            raw_sents = sent_tokenizer.tokenize(file_content)
            processed_sents = []
            raw_sents_with_quotes = []
            for sent in raw_sents:
                # pre-processing for data to used as the input of sumbasic algorithm
                # lemmatization
                wnl = WordNetLemmatizer()
                lemmatized_word_token = []
                nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sent))
                wn_tagged = map(lambda x: (x[0], self.nltk2wn_tag(x[1])), nltk_tagged)

                for word, tag in wn_tagged:
                    if tag is None:
                        lemmatized_word_token.append(word)
                    else:
                        lemmatized_word_token.append(wnl.lemmatize(word, tag))

                # lemmatized_word_token = [wnl.lemmatize(i, self.nltk2wn_tag(j)) for i, j in pos_tag(word_tokenize(sent))]

                # remove stop words
                stop_words = set(stopwords.words('english'))
                remove_stop_word_token = [w for w in lemmatized_word_token if not w.lower() in stop_words]


                # remove punctuation
                punctuation = set(string.punctuation)
                remove_nonassci_word_and_nonascii_list = []
                for word in remove_stop_word_token:
                    remove_nonassci_word_and_nonascii = ''.join(i for i in word if i not in punctuation and ord(i) < 128)
                    remove_nonassci_word_and_nonascii_list.append(remove_nonassci_word_and_nonascii)

                # to lower case
                lower_case = [word.lower() for word in remove_nonassci_word_and_nonascii_list]

                # processed_sent
                processed_sent = ' '.join(lower_case)

                processed_sents.append(processed_sent)

                # pre-processing for the output of the sumbasic algorithm
                # adding quotation to middle-quote sentence to form raw data
                raw_quote = sent
                if sent in quote_sents:
                    raw_quote = "\"" + sent + "\""

                raw_sents_with_quotes.append(raw_quote)


            se_processed = pd.Series(processed_sents)
            sent_doc_df['processed'] = se_processed.values

            se_raw = pd.Series(raw_sents_with_quotes)
            sent_doc_df['raw'] = se_raw.values

            # append to the
            self.sent_df = self.sent_df.append(sent_doc_df)

            word_list = []
            for sent in processed_sents:
                word_list = word_list + word_tokenize(sent)

            word_counts = Counter(word_list)

            # add to the old dictionary
            for key, value in word_counts.items():
                self.vocab_dict[key] = self.vocab_dict.get(key, 0) + word_counts[key]


    # version takes one of: orig, best-avg and simplified.
    def sum_basic(self):
        if self.version == 'best-avg':
            # step 1: get probability for each word
            total_word_freq = sum(self.vocab_dict.values())

            for word, freq in self.vocab_dict.items():
                self.vocab_prob_dict[word] = freq/total_word_freq

            while self.summary_length < 100:
                # step 2: calculate weight for each sentence
                self.sent_df['weight'] = self.sent_df.apply(self.calculate_weight, axis=1)

                # step 3: skipped

                # pick the best avg sentence
                self.sent_df = self.sent_df.sort_values(['weight'], ascending=[0])
                self.sent_df = self.sent_df.reset_index(drop=True)
                summary_sent = self.sent_df.iloc[0]

                # my design: remove the selected  sentence
                self.sent_df = self.sent_df.drop(0)

                # step 4: update the probability
                summary_processed_sent = summary_sent['processed']
                summary_processed_words = word_tokenize(summary_processed_sent)
                for word in summary_processed_words:
                    self.vocab_prob_dict[word] = self.vocab_prob_dict.get(word) * self.vocab_prob_dict.get(word)

                # step 5: check summary length, then go back to step 2 if length is not reached
                summary_raw_sent = summary_sent['raw']
                summary_raw_words = word_tokenize(summary_raw_sent)
                if self.summary == '':
                    self.summary = self.summary + summary_raw_sent
                else:
                    self.summary = self.summary + ' ' + summary_raw_sent
                self.summary_length = self.summary_length + len(summary_raw_words)
            return

        elif self.version == 'orig':
            # step 1: get probability for each word
            total_word_freq = sum(self.vocab_dict.values())

            for word, freq in self.vocab_dict.items():
                self.vocab_prob_dict[word] = freq / total_word_freq

            while self.summary_length < 100:
                # step 2: calculate weight for each sentence
                self.sent_df['weight'] = self.sent_df.apply(self.calculate_weight, axis=1)

                # step 3: pick the best scoring sentence that contains the highest probability word
                # find highest probability word
                highest_probability_word = max(self.vocab_prob_dict.items(), key=operator.itemgetter(1))[0]
                # print("highest_probability_word:" + highest_probability_word)
                # print("with probability: " + str(self.vocab_prob_dict[highest_probability_word]))
                self.sent_df['contain_word'] = \
                    self.sent_df.apply(lambda row: highest_probability_word in word_tokenize(row['processed']), axis=1)
                self.sent_df = self.sent_df.sort_values(['contain_word', 'weight'], ascending=[0, 0])
                self.sent_df = self.sent_df.reset_index(drop=True)
                summary_sent = self.sent_df.iloc[0]

                # my design: remove the sentence
                self.sent_df = self.sent_df.drop(0)

                # step 4: update the probability
                summary_processed_sent = summary_sent['processed']
                summary_processed_words = word_tokenize(summary_processed_sent)
                for word in summary_processed_words:
                    self.vocab_prob_dict[word] = self.vocab_prob_dict.get(word) * self.vocab_prob_dict.get(word)

                # step 5: check summary length, then go back to step 2 if length is not reached
                summary_raw_sent = summary_sent['raw']
                # print("add sentence: " + summary_raw_sent)
                summary_raw_words = word_tokenize(summary_raw_sent)
                # print("tokenized sentence: " + str(summary_raw_words))
                if self.summary == '':
                    self.summary = self.summary + summary_raw_sent
                else:
                    self.summary = self.summary + ' ' + summary_raw_sent
                self.summary_length = self.summary_length + len(summary_raw_words)
                # print("summary length is now: " + str(self.summary_length))
            return
        elif self.version == 'simplified':
            # step 1: get probability for each word
            total_word_freq = sum(self.vocab_dict.values())

            for word, freq in self.vocab_dict.items():
                self.vocab_prob_dict[word] = freq / total_word_freq

            while self.summary_length < 100:
                # step 2: calculate weight for each sentence
                self.sent_df['weight'] = self.sent_df.apply(self.calculate_weight, axis=1)

                # step 3: pick the best scoring sentence that contains the highest probability word
                # find highest probability word
                highest_probability_word = max(self.vocab_prob_dict.items(), key=operator.itemgetter(1))[0]
                # print("highest_probability_word:" + highest_probability_word)
                # print("with probability: " + str(self.vocab_prob_dict[highest_probability_word]))
                self.sent_df['contain_word'] = \
                    self.sent_df.apply(lambda row: highest_probability_word in word_tokenize(row['processed']), axis=1)
                self.sent_df = self.sent_df.sort_values(['contain_word', 'weight'], ascending=[0, 0])
                self.sent_df = self.sent_df.reset_index(drop=True)
                summary_sent = self.sent_df.iloc[0]

                # my design: remove the sentence
                self.sent_df = self.sent_df.drop(0)

                # step 4: skipped

                # step 5: check summary length, then go back to step 2 if length is not reached
                summary_raw_sent = summary_sent['raw']
                summary_raw_words = word_tokenize(summary_raw_sent)
                if self.summary == '':
                    self.summary = self.summary + summary_raw_sent
                else:
                    self.summary = self.summary + ' ' + summary_raw_sent
                self.summary_length = self.summary_length + len(summary_raw_words)
            return
        else:
            return

    def calculate_weight(self, row):
        processed_sent = row['processed']
        words = word_tokenize(processed_sent)
        word_count = len(words)

        weight_sum = 0
        for word in words:
            word_prob = self.vocab_prob_dict.get(word)
            weight_sum = weight_sum + word_prob

        if word_count != 0:
            return weight_sum/word_count
        else:
            return 0

    def nltk2wn_tag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

def main(algo, input_files):
    # python ./sumbasic.py simplified ./docs/doc1-*.txt > simplified-1.txt
    if algo == 'simplified' or algo == 'orig' or algo == 'best-avg':
        sumbasic = SumBasic(algo, input_files)
        sumbasic.pre_process_data()
        sumbasic.sum_basic()
        # file = open("write_test.txt", 'w')
        # file.write(sumbasic.summary)
        # file.close()
        print(sumbasic.summary)
        # print("summary length is: " + str(sumbasic.summary_length))
    elif algo == 'leading':
        sumleading = BaseLine(algo, input_files)
        sumleading.sum_leading()
        print(sumleading.summary)
        # print("summary length is: " + str(sumleading.summary_length))


if __name__ == '__main__':
    # python ./sumbasic.py simplified ./docs/doc1-*.txt > simplified-1.txt
    arguments = sys.argv[1:]
    if len(arguments) < 2:
        print("argument error: please follow the following execution example to run the code")
        print("python ./sumbasic.py simplified ./docs/doc1-*.txt")
    elif arguments[0] not in {'simplified', 'best-avg', 'orig', 'leading'}:
        print("argument error: the first argument needs to be one of 'simplified', 'best-avg', 'orig', 'leading'")
    else:
        main(arguments[0], arguments[1])

