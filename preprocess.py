import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle

body = pd.read_csv('fnc-1/train_bodies.csv')
stances = pd.read_csv('fnc-1/train_stances.csv')

body = body.sort_values(by='Body ID')

def create_TF_vocab():
    '''Creates a vocabulary of top 5000 words from training set.'''

    cv = CountVectorizer(stop_words='english')
    cv_fit = cv.fit_transform(body['articleBody'])
    y = cv_fit.toarray()
    y_sum = y.sum(axis=0)
    print(y_sum)
    y_sum = list(y_sum)
    words = cv.get_feature_names()

    freq = []
    for i in range(len(words)):
        freq.append((words[i],y_sum[i]))

    freq.sort(key=lambda x:x[1])
    top_words_vocab = freq[-5000:]
    len(top_words_vocab)
    vocab = [word for word,freq in top_words_vocab]
    return vocab

vocab = create_TF_vocab()
cv_bow = CountVectorizer(stop_words='english',vocabulary=vocab)
cv_bow.fit(body['articleBody'])
#cv_feat = cv_bow.fit_transform(body['articleBody'])

with open('count_vectorizer1.pk','wb') as fin:
    pickle.dump(cv_bow,fin)

#count_vectorizer = pickle.load(open('count_vectorizer1.pk','rb'))

def create_TFIDF_vocab():
    '''Creates a vocabulary of top 5000 words by combining training and test sets. '''

    test_body = pd.read_csv('fnc-1/test_bodies.csv')
    test_body = test_body.sort_values(by='Body ID')
    cv_test = CountVectorizer(stop_words = 'english')
    concat_data = pd.concat([body['articleBody'],test_body['articleBody']] ,axis=0)
    cv_test_fit = cv_test.fit_transform(concat_data)
    test_term_freq = cv_test_fit.toarray()
    test_term_freq_sum = test_term_freq.sum(axis=0)
    test_term_freq_sum = list(test_term_freq_sum)

    tfidf_vocab = []
    idf_words = cv_test.get_feature_names()
    for i in range(len(idf_words)):
        tfidf_vocab.append((idf_words[i],test_term_freq_sum[i]))

    tfidf_vocab.sort(key=lambda x:x[1])
    tfidf_vocab = tfidf_vocab[-5000:]
    idf_vocab = [word for word,fr in tfidf_vocab]

    return tfidf_vocab

tfidf_vec = TfidfVectorizer(vocabulary=idf_vocab)
tfidf_vec.fit(concat_data)
#tfidf_features = tfidf_vec.fit_transform(concat_data)


pickle.dump(tfidf_vec,open('tfidf_vectorizer.pk','wb'))

#tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pk','rb'))



train_stance = pd.read_csv('fnc-1/train_stances.csv')
test_stance = pd.read_csv('fnc-1/test_stances_unlabeled.csv')
train_stance = train_stance.sort_values(by='Body ID')
stance_features = cv_bow.transform(train_stance['Headline'])
print(stance_features.shape)

pickle.dump(stance_features,open('headline_transformed.pk','wb'))


stance_bids = list(train_stance['Body ID'])
unique_stance = set(stance_bids)
count = []
for x in unique_stance:
    count.append((x,stance_bids.count(x)))

#print(count)

#Combining the dataframes
join_df = pd.merge(train_stance,body_copy,on='Body ID')

def calculate_tfidf_score(row):
    ''' Calculates the similarity between the headline and body by using cosine similarity of tfidf scores.'''

    headline = row['Headline']
    #print(headline)
    body = row['articleBody']
    #print(body)
    headline_features = tfidf_vec.transform([headline]).toarray()
    body_features = tfidf_vec.transform([body]).toarray()
    #print(body_features)
    #print(body_features.shape)
    #print(headline_features)
    #print(headline_features.shape)
    dot = np.dot(headline_features,body_features.T)
    #print(dot)
    return dot[0][0]

# Apply the operation
join_df['TFIDF'] = join_df.apply(calculate_tfidf_score,axis=1)


def create_input(row):
    ''' Creates the required input by combining term frequencies and tfidf similarity scores.'''

    headline = row['Headline']
    #print(headline)
    body = row['articleBody']
    #print(body)
    headline_features = cv_bow.transform([headline]).toarray()
    body_features = cv_bow.transform([body]).toarray()
    #print(body_features.shape)
    tfidf_score = np.array(row['TFIDF'])
    #print(tfidf_score.shape)
    input_feature = np.column_stack((headline_features,tfidf_score,body_features))
    #print(input_feature.shape)
    return input_feature[0]

# Apply the operation
join_df['input'] = join_df.apply(create_input,axis=1)

# Pickling the dataframe
join_df.to_pickle('./inputdf')
copy = pd.read_pickle('./inputdf')

# Converting stance label from categorical to numerical
# copy['Stance'] = copy['Stance'].astype('category')
# cat_columns = copy.select_dtypes(['category']).columns
# copy[cat_columns] = copy[cat_columns].apply(lambda x:x.cat.codes)

def convert_stance(row):
    stance = row['Stance']
    print(stance)
    if stance == 'agree':
        return 0
    elif stance == 'disagree':
        return 1
    elif stance == 'discuss':
        return 2
    elif stance == 'unrelated':
        return 3
copy['numerical_stance'] = copy.apply(convert_stance,axis=1)

copy.to_pickle('./updated_df')
