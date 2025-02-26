# Bizparse - An NLP Library for Business Documents

### Overview

This project aimed to take in business related documents, the examples used are all 10Ks, and provide 
multiple insightful visualizations that can help someone understand large scale trends in these documents.
This is not intended as a replacement for the understanding gained by carefully reading and thinking about 
the content of these documents, rather it's purpose is to one, provide insight quicker than reading an 
entire document and two, demonstrate large scale trends that aren't readily available to the human eye. 
This posed a creative challenge, to try and come up with insights into some incredibly scrutinized 
documents, however I believe myself and the three other peers of mine that worked on this came up with 
interesting visualizations that provide value to people with any level of experience with business lingo.

### Figures

Starting off we have a comparative tool, which works by measuring the individual counts of all words in 
two documents, then subtracting one of the wordcounts from the other and plotting the most common words
plus vice versa. This shows you the main difference in what is being discussed in two different documents.
There were several interesting results of this tool being used in different cases. The example below
showcases the difference between Nvidia and AMD, two competing companies who have had drastically different
performances in the recent rise of AI despite the fact that they both manufacture graphics cards. The 
things being discussed in their respective 10K filings could be indicators of why this happened. Nvidia 
places more emphasis on the words impact, demand, and networking while AMD contained far more occurences of 
agreement, company, broadcom, and technology. Nvidia seemed to be focused on larger scale external topics 
and less focused on the internal workings of their company.

![better_comp_bar](https://github.com/user-attachments/assets/feff4a94-9826-4bcf-b451-0fd55a6619b3)

Next we created a bar chart which analyzed the sentiment score of sentences conatining different words. 
We created a custom text parser which split the text into individual sentences, then search for the 
desired keywords and when they were found calculated a sentiment score ranging from -1 to 1 of that sentence
then averaged the score of all sentences where that specific word occurred. In the example below, again of
Nvidia and AMD, there are significant differences in the sentiments surrounding cost, competitor, and 
profit. What's particulary interesting is that AMD talks much more positively about profit, when it has 
certainly had a much less profitable year that Nvidia.

![Figure_1](https://github.com/user-attachments/assets/7897b739-e531-4f46-b164-3ab0064f41fd)

Following this is a general sentiment tool, that graphs the distribution of each word's sentiment in each
document. You can see that distrubtuions are fairly similar across these documents with Etrade being the 
most skewed positively.  

![Figure_5](https://github.com/user-attachments/assets/399f5e4c-8874-47e9-b03d-2c5e1f419b64)

Next up is a wordcloud generated for each document. This is our most general tool, solely conveyeing this
biggest picture words of the document, with the size of each word being scaled based on frequency. 

![Figure_3](https://github.com/user-attachments/assets/d155ab90-a8e1-484f-bad7-079311f297fe)

This figure is a sankey diagram generated using plotly. It links documents to words with the link thickness
representing relative frequency. Our plot displays words that occur more than a certain threshold within a 
document, which for this figure is 250. This graph is useful for seeing what overlap there is between 
certain companies, which in this example in nothing of note, but can demonstrate in other cases interesting
patterns.

![newplot](https://github.com/user-attachments/assets/883f0672-b07d-4ff1-9777-9d659ae02bbf)

Lastly we have a bar chart that demonstrates the most common words among all of the documents that have 
been loaded in. This was used on our part to gain understanding of a singular company, or an industry on the
whole. 

![Figure_2](https://github.com/user-attachments/assets/d594a7a9-744f-4ac1-9089-e83f73188832)

These are all the current capabilities of the library. WIth such an open ended task there are so many 
possibilities, which is why I want to come back to this project in the future. 

Project Members: Samuel Baldwin, Jack Carroll, Jeffery Krapf, Michael Masseide
