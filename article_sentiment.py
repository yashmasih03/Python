#Get the sentiment of articles from the internet

#pip install newspaper3k

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

text = article.summary

#print text
print(text)

obj = TextBlob(text)

#returns the sentiment of text
#by returning a value between -1.0 and 1.0
sentiment = obj.sentiment.polarity
print(sentiment)

if sentiment == 0:
  print('The article is neutral')
elif sentiment > 0:
  print('The article is positive')
else:
  print('The article is negative')
