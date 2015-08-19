import pandas as pd
import numpy

# load train data
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# print train.shape
# print train.columns.values
# print train["review"][1]


###################### Cleanup review text ###################
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def clean_review(raw):
    # remove HTML
    review_text = BeautifulSoup(raw).get_text()
    # remove no-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text )
    words = review_text.lower().split()
    # remove stopwords
    stops = stopwords.words("english")
    meaningful_words = [w for w in words if not w in stops]
    
    return " ".join(meaningful_words)

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

for i in range(num_reviews):
    # clean reviews
    clean_train_reviews.append( clean_review( train["review"][i] ) )


################# Feature Extraction ########################
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Using TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(train_data_features)


#################### Random Forest Classifier ##################
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( X_train_tfidf, train["sentiment"] )


######################## Create a submission ##################
# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in range(num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review_text = clean_review( test["review"][i] )
    clean_test_reviews.append( clean_review_text )

# Get a bag of words for the test set, and convert to a numpy array
# test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = vectorizer.transform(clean_test_reviews)
tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.fit_transform(test_data_features)

# test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
# result = forest.predict(test_data_features)
result = forest.predict(X_test_tfidf)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model_tfidf_RF.csv", index=False, quoting=3 )
