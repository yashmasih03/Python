from sklearn.metrics.pairwise import cosine_similarity

text = ["Amazing Spiderman Amazing","Spiderman Spiderman Amazing"]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

#Print the count of each word within each text in the list 
print(cv.get_feature_names())
print(count_matrix.toarray())


#Print the similarity scores
print("\nSimilarity Scores:")
print(cosine_similarity(count_matrix)) 
