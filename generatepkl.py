import numpy as np
import pandas as pd
import ast
import pickle

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from sklearn.metrics.pairwise import cosine_similarity

movies= pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')

movies= movies.merge(credits,on='title')

movies= movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace= True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres']= movies['genres'].apply(convert)
movies['keywords']= movies['keywords'].apply(convert)

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast']= movies['cast'].apply(convert3)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew']= movies['crew'].apply(fetch_director)
movies['overview']=movies['overview'].apply(lambda x:x.split())

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

df = movies[['movie_id','title','tags']]
df['tags']= df['tags'].apply(lambda x:" ".join(x))

df['tags']= df['tags'].apply(lambda x:x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors= cv.fit_transform(df['tags']).toarray()

ps = PorterStemmer()

def stem(text):
    y= []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)

df['tags']= df['tags'].apply(stem)

similarity= cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True, key=lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = df[df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(df.iloc[i[0]].title)

pickle.dump(df,open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))