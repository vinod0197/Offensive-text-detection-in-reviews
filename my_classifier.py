  import re
    import string
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    porter = PorterStemmer()
    
    lower_case_text = text.translate(None, string.punctuation).lower()
    words_list = word_tokenize(lower_case_text)
    text2 = []
    for word in words_list:
        text0 = word.decode('ascii', 'ignore')
        text1 = regex.sub(u'', text0)
        if not text1 == u'':
            if not text1 in stopwords.words('english'):
                text2.append(porter.stem(text1))
    return text2

data['comment_clean'] = data['Comment'].apply(transform_text)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

def split_into_lemmas(comments):
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)
    analyze = bigram_vectorizer.build_analyzer()
    return analyze(comments)

vectorizer = CountVectorizer(analyzer=split_into_lemmas,stop_words='english',strip_accents='ascii').fit(data['comment_string'])
text_transformed = vectorizer.transform(data['comment_string'])
tfidf_transformer = TfidfTransformer().fit(text_transformed)
tfidf_transformed_text = tfidf_transformer.transform(text_transformed)
#def calculate_goodNdbadWord_ratio(text):
def calculate_bad_word_ratio(text):
    #print text
    num_words = len(text) if len(text) > 0 else 1
    #print num_words
    num_bad_words = 0
    #num_good_words = 0
    bad_wrd_ratio = 0
    #good_wrd_ratio = 0
    for word in text:
        if word in bad_words:
            num_bad_words += 1
        #if word in good_words:
            #num_good_words += 1
    bad_wrd_ratio = float(num_bad_words)/float(num_words)
    #good_wrd_ratio = float(num_good_words)/float(num_words)
    #print num_bad_words
    #ratio = [bad_wrd_ratio, good_wrd_ratio]
    #print ratio
    #return ratio
    return bad_wrd_ratio

data['bad_word_ratio'] = data['comment_clean'].apply(calculate_bad_word_ratio)
#ratio = data['comment_clean'].apply(calculate_goodNdbadWord_ratio)
#data['bad_word_ratio'] = ratio[0]
#data['good_word_ratio'] = ratio[1]
#print data[data['bad_word_ratio'] > 0]
data