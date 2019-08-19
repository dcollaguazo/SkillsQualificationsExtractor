from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
import re

def pre_process(text):
    # lowercase
    text=text.lower()
    #remove tags
    text=re.sub("<!--?.*?-->","",text)    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)

    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_list = []
    for word in text.split():
    	if word not in stop_words:
    		word_list.append(word)
    return text


if __name__ == "__main__":
    csv_filepath = 'tors_with_language.csv'
    df = pd.read_csv(csv_filepath, header=1, keep_default_na=False)
    df = df.loc[df.iloc[:,1] == 'en']
    df = df.iloc[:,0]

    df = df.apply(pre_process)
    content = df.tolist()
    cv=CountVectorizer(max_df=0.90)
    word_count_vector=cv.fit_transform(content)
    print(list(cv.vocabulary_.keys())[:200])