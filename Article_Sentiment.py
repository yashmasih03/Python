
#Get the sentiment of articles from the internet

#pip install newspaper3k
#pip install textblob
#pip install nltk

#Import the libraries
from textblob import TextBlob
import nltk
from newspaper import Article

#Get the article
url = 'https://everythingcomputerscience.com/'
article = Article(url)

# Do some NLP
article.download() #Downloads the link’s HTML content
article.parse() #Parse the article
nltk.download('punkt')#1 time download of the sentence tokenizer
article.nlp()#  Keyword extraction wrapper

#Get the summary of the article
text = article.summary

#print text
print(text)

#Create Text Blob Object
#NOTE: You can treat TextBlob objects as if they were Python strings that learned how to do Natural Language Processing.
obj = TextBlob(text)

#returns the sentiment of text
#by returning a value between -1.0 and 1.0
sentiment = obj.sentiment.polarity
print(sentiment)

#Print if the article was neutral, positive, or negative
if sentiment == 0:
  print('The text is neutral')
elif sentiment > 0:
  print('The text is positive')
else:
  print('The text is negative')
