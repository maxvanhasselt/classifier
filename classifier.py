from sklearn.feature_extraction.text import TfidfVectorizer
#from stop_words import get_stop_words
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pandas as pd
import numpy as np

#Set random seed zodat de trainset en testset altijd het zelfde zijn.
np.random.seed(19932005)

#laad de labeledTrainData en splits in train en test
df = pd.read_csv('labeledTrainData.tsv', sep='	')
msk = np.random.rand(len(df)) < 0.9

trainDf = df[msk]
testDf = df[~msk]

sentiments = ['0','1']


# Instantieer TF-IDF Vectorizer
#	stop_words: de stopwords die moeten worden overgeslagen
#	sublinear_tf = True: gebruik sublineare functie, 
#						zodat woorden die te vaak gebruikt 
#						worden minder zwaar wegen.
#	use_idf = True: Gebruik inverse document frequency zodat
#					woorden die minder vaak gebruikt worden
#					zwaarder wegen.
vectorizer = TfidfVectorizer(
							stop_words = 'english',
							sublinear_tf = True,
							use_idf = True)

train_vectors = vectorizer.fit_transform(trainDf['review'])
test_vectors = vectorizer.transform(testDf['review'])

labels = trainDf['sentiment'].values

# Fit het Naive Bayes model
clf = MultinomialNB().fit(train_vectors,labels)

# Voorspel het sentiment van de test set.
predicted = clf.predict(test_vectors)

# Bereken de precisie
accuracy = np.mean(predicted == testDf['sentiment'])

print('Accuracy: ', accuracy)

print(metrics.classification_report(testDf['sentiment'], predicted,
									target_names=sentiments))

# maak confusion matrix
confusion_matrix = metrics.confusion_matrix(testDf['sentiment'], predicted)
print(confusion_matrix)

# test met eigen reviews. 
testReviews ={'review': ['This movie was very innovative.', 
						'I hated this movie, would not recommend.', 
						'You should watch this movie if you hate fun.']}
testReviewsDf = pd.DataFrame(data=testReviews)

testReviewVector = vectorizer.transform(testReviewsDf['review'])

predictions = clf.predict(testReviewVector)



for prediction in predictions:
	if(prediction == 1):
		print('positief')
	else:
		print('negatief')



