import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# Read the dictionary from the text file
with open('example_output.txt', 'r') as file:
    dictionary = ast.literal_eval(file.read())


# wordcloud = WordCloud(
#     width=800,
#     height=400,
#     background_color='white',
#     colormap='viridis'
# ).generate_from_frequencies(dictionary)
#
# # Display the word cloud
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.title("Word Cloud of Word Frequencies")
# plt.show()
#

sentiment_data = []
for word, count in dictionary.items():
    polarity = TextBlob(word).sentiment.polarity
    if polarity != 0:
        sentiment_data.extend([polarity] * count)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(sentiment_data, bins=10, edgecolor='black')
plt.title("Histogram of Word Sentiments")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Number of Words")
plt.show()

