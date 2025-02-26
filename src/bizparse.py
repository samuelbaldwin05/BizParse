"""

file: bizparse.py

Description: A reusable library for text analysis and comparison
In theory, the framework should support any collection of texts
of interest (though this might require the implementation of some
custom parsers.)

The core data structure:

Input: "A" --> raw text,  "B" --> another text

Extract wordcounts:
        "A" --> wordcounts_A,   "B" --> wordcounts_B, ......

What get stored:

        "wordcounts"  --->   {"A" --> wordcounts_A,
                              "B" --> wordcounts_B, etc.}

        e.g., dict[wordcounts][A] --> wordcounts_A



"""

from collections import defaultdict, Counter
import random as rnd
import matplotlib.pyplot as plt
import PyPDF2
import re
import sankey2 as sk
import pandas as pd
from wordcloud import WordCloud
from sankey2 import make_sankey
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# NEW PACKAGE USED need "conda install PyPDF2"
# NEW PACKAGE USED "conda install nltk"

class Bizparse:
    def __init__(self):
        """ Constructor

        datakey --> (filelabel --> datavalue)
        """
        self.data = defaultdict(dict)
        self.stopwords = []
        self.keywords = []

    """
    Idea for a viz: do operations with counter, subtract them from each other and after that see what 
    the diff is. Examine this change.
    """

    def cleaner(self, text):
        """
        Piazza recommended separate functions for cleaning/parsing so we do it.
        This function takes in a string of text and a list of stop words and cleans
        the text, returning it as a list of words.
        """
        new_text = ""
        # replace newlines w spaces
        filtered_text = re.sub(r"\n", " ", text)
        # Replace apostrophes with nothing
        filtered_text = re.sub(r"'", "", filtered_text)
        # remove weird characters
        filtered_text = re.sub(r"[^\x20-\x7E]", "", filtered_text)
        # remove extra spaces
        filtered_text = re.sub(r"\s+", " ", filtered_text)
        # Remove punctuation
        filtered_text = re.sub("[^-9A-Za-z ]", "", filtered_text)
        # make lowercase
        new_text += "".join([i.lower() for i in filtered_text])
        lst = new_text.split(" ")
        lst = [word for word in lst if word not in self.stopwords and word != ""]
        return lst
    def sentence_cleaner(self, text):
        """
        Function that cleans the txt of the pdf and returns a lsts of sentences
        """
        new_text = ""
        filtered_text = re.sub(r"\n", " ", text)
        filtered_text = re.sub(r"'", "", filtered_text)
        filtered_text = re.sub(r"[^\x20-\x7E]", "", filtered_text)
        filtered_text = re.sub(r"\s+", " ", filtered_text)
        filtered_text = re.sub(r"[^A-Za-z0-9. ]", "", filtered_text)
        new_text += "".join([i.lower() for i in filtered_text])
        lsts = re.split(r'(?<!\w\.\w)(?<![A-Z][a-z])\.\s', new_text)
        print(lsts)
        final = []
        for sentence in lsts:
            sentence = sentence.split(" ")
            total = ([word for word in sentence if word not in self.stopwords and word != ""])

            total = " ".join(total)
            final.append(total)
        return final

    def default_parser(self, filename):
        """
        Parse a standard text file and produce
        extract data results in the form of a dictionary.
        """

        # SEC filings most commonly available for download as a PDF, so that is our default parser
        text = ""
        with open(filename, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(filename)
            for i in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[i]
                raw_text = page.extract_text()
                text += "".join([i for i in raw_text])

        clean_text = self.cleaner(text)

        c = Counter(clean_text)

        results = {'wordcount': c, 'numwords': c.total()}

        return results
            
    def load_key_words(self, keywords_file):
        """
        Load a list of keywords and attatchs them to the object. Keywords will be used
        in sentence parser
        """
        words = []
        with open(keywords_file, "r") as infile:
            for line in infile:
                words.append(line.strip().lower())
        self.keywords = words

    def load_stop_words(self, stopwords_file):
        """
        Load in our stopwords file and return it as a list of words. Assign this list to the
        stopwords attribute. Our file is taken from The University of Notre Dame's accounting
        and finance department, and catered slightly from NLTK's default list for financial documents.
        WORDS ADDED: A,
        """
        words = []
        with open(stopwords_file, "r") as infile:
            for line in infile:
                words.append(line.strip().lower())
        self.stopwords = words

    def sentence_parser(self, filename):
        """
        Parses through the pdf while extracting the data as sentences. Then performs sentiment analysis on sentences
        which have certain key words within it. Builds a dictionary that contains key words as keys and an average sentiment of sentences that
        contain said keywor

        """
        lemmatizer = WordNetLemmatizer()
        sia = SentimentIntensityAnalyzer()
        polarity_results = {}
        text = ""

        with open(filename, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(filename)
            for i in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[i]
                raw_text = page.extract_text()
                text += "".join([i for i in raw_text])

        sentences = self.sentence_cleaner(text)
        key_words = self.keywords
        key_words = [lemmatizer.lemmatize(word) for word in key_words]
        print(key_words)

        lemmatized_sentences = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
            lemmatized_sentences.append(" ".join(lemmatized_tokens))
        for sentence in lemmatized_sentences:
            for keyword in key_words:
                if re.search(rf'\b{keyword}\b', sentence, re.IGNORECASE):
                    if keyword not in polarity_results.keys():
                        sentiment = sia.polarity_scores(sentence)
                        polarity_results[keyword] = round(sentiment["compound"], 2)
                    else:
                        sentiment = sia.polarity_scores(sentence)
                        polarity_results[keyword] = round((sentiment["compound"] + polarity_results[keyword]) / 2, 2)
        polarity_keys = polarity_results.keys()
        for key in key_words:
            if key not in polarity_keys:
                polarity_results[key] = 0
        print(polarity_results)
        return polarity_results

    def load_text(self, filename, label=None, parser=None):
        """ Register a document with the framework.
        Extract and store data to be used later by
        the visualizations """
        if parser is None:
            results = self.default_parser(filename)
        if parser == "sentence_parser":
            results = self.sentence_parser(filename)

        if label is None:
            label = filename
        
        if parser is None:
            for k, v in results.items():
                self.data[k][label] = v
        if parser == "sentence_parser":
            for k, v in results.items():
                self.data[label][k] = v
                    
    def plot_sentence_parser_data(self, label_lst, color="blue"):
        """
        Plotting out results from the sentence sentiment analyzer. Plot will be in the form of subplots 
        """
        num_dicts = len(label_lst)
        
        fig, axes = plt.subplots(num_dicts, 1, figsize=(10, 6 * num_dicts), constrained_layout=True)
        
        if num_dicts == 1:
            axes = [axes]
        idx = 0
        for label in label_lst:
            data_dict = self.data[label]
            words = list(data_dict.keys())
            percentages = list(data_dict.values())
            colors = ['green' if percent > 0 else 'red' for percent in percentages]

            ax = axes[idx]
            ax.bar(words, percentages, color=colors)
            
            ax.set_title(f"{label} key word sentiment analysis", fontsize=14)

            ax.set_xlabel("Key words")
            ax.set_ylabel("Sentiment Percentange")
            ax.set_xticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45, ha="right")
            
            for i, v in enumerate(percentages):
                ax.text(i, v + 0.01, v, ha="center", fontsize=9)
            idx += 1
        plt.show()

    def sankey(self, filter):
        """ Creates sankey that connects company to words in the article that are repeated
        more times than the filter
        """
        final_df = pd.DataFrame()
        wordcounts = self.data["wordcount"]
        for text in wordcounts.keys():
            df = pd.DataFrame(list(wordcounts[text].items()), columns=["word", "count"])
            df['text'] = text
            df = df[df['count'] > filter]
            final_df = pd.concat([final_df, df], ignore_index=True)
        make_sankey(final_df, 'text', 'word', vals='count', filter=filter)

    def commonwords_barchart(self, n=10):
        """ Aggregate word counts from all articles into a single dictionary
        and plot a bar chart of the most common words.
        """
        # Empty counter to sum articles
        summed_counts = Counter()

        # Just use word count data
        wordcounts = self.data["wordcount"]
        # Iterate through each article's word count dictionary and aggregate
        for company, counter in wordcounts.items():
            summed_counts.update(counter)

        # Get n most common words
        most_common_words = summed_counts.most_common(n)

        # Split words and counts for plotting
        words, counts = zip(*most_common_words)

        # Create bar chart and plot
        plt.figure(figsize=(10, 6))
        plt.bar(words, counts, edgecolor='black')
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.title(f"Top {n} Most Common Words Across Articles")
        plt.xticks(rotation=90)
        plt.tight_layout()

        plt.show()

    def wordcloud(self):
        """ Create a word cloud for each company's word counter in self.data.
        """
        # Count number of companies to fix subplot axes
        num_companies = len(self.data["wordcount"])

        # Set up subplots
        fig, axes = plt.subplots(nrows=num_companies, figsize=(10, 8 * num_companies))

        # If 1 company set axes as a list to work with subplots
        if num_companies > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Get data of only word counter
        wordcounts = self.data["wordcount"]
        for idx, (company, counter) in enumerate(wordcounts.items()):
            # Create the word cloud from the word frequencies dictionary
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  colormap='viridis').generate_from_frequencies(counter)

            # Plot word cloud on subplot
            ax = axes[idx]
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f"Word Cloud for {company}")

        # Show plots
        fig.subplots_adjust(hspace=.5)
        plt.show()

    def sentiment_histogram(self, bins=10):
        """Create sentiment histograms for each companies word counts in self.data
        Make subplot by iterating through companies
        """
        # Count number of companies to fix subplot axes
        num_companies = len(self.data["wordcount"])

        # Set up subplots
        fig, axes = plt.subplots(nrows=num_companies, figsize=(10, 6 * num_companies))

        # If 1 company set axes as a list to work with subplots
        if num_companies > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Get data of only word counter
        wordcounts = self.data["wordcount"]
        # Enumerate through each company to get
        for idx, (company, counter) in enumerate(wordcounts.items()):
            sentiment_data = []
            # Iterate through each word in the dictionary
            for word, count in counter.items():
                polarity = TextBlob(word).sentiment.polarity
                # Only include words with non-zero polarity
                if polarity != 0:
                    sentiment_data.extend([polarity] * count)

            # Create subplot
            ax = axes[idx]
            ax.hist(sentiment_data, bins=bins, edgecolor='black')
            ax.set_title(f"Sentiment Histogram for {company}")
            ax.set_xlabel("Sentiment Polarity")
            ax.set_ylabel("Number of Words")

        # Space plots out better
        fig.subplots_adjust(hspace=.3)

        # Show plots
        plt.show()

    def comparison_bar(self, name1, name2):
        """
        Take in the names of two different companies, and generate two different plots by subtracting
        their counters and graphing the most frequent words after.
        """
        counter1 = self.data["wordcount"][name1] - self.data["wordcount"][name2]
        counter2 = self.data["wordcount"][name2] - self.data["wordcount"][name1]

        c1_top10 = dict(sorted(counter1.items(), key=lambda item: item[1], reverse=True)[:20])
        c2_top10 = dict(sorted(counter2.items(), key=lambda item: item[1], reverse=True)[:20])

        c1_x = c1_top10.keys()
        c1_y = c1_top10.values()
        c2_x = c2_top10.keys()
        c2_y = c2_top10.values()


        fig, ax = plt.subplots(1, 2, figsize = (12, 6))
        ax[0].bar(c1_x, c1_y, color = "skyblue")
        ax[0].set_xlabel("Words")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title(f"Most common words from difference of {name1} and {name2}")
        ax[0].tick_params(axis = "x", rotation = 90)

        ax[1].bar(c2_x, c2_y, color="orange")
        ax[1].set_xlabel("Words")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title(f"Most common words from difference of {name2} and {name1}")
        ax[1].tick_params(axis = "x", rotation = 90)
        plt.tight_layout()
        plt.show()


