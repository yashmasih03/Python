#import the library sklearn to use the cosine similarity function
from sklearn.metrics.pairwise import cosine_similarity

#A list of text
text = ["Amazing Spiderman Amazing","Spiderman Spiderman Amazing"]

#Create a Count Vector object to get a count of each word in the text
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

#Print the count of each word within each text in the list 
print(cv.get_feature_names())
print(count_matrix.toarray())


#Print the similarity scores
print("\nSimilarity Scores:")
print(cosine_similarity(count_matrix)) 
