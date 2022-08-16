import os
from turtle import distance
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    os.system('pip install sklearn')
    from sklearn.metrics.pairwise import cosine_similarity

try:
    import numpy as np
except ModuleNotFoundError:
    os.system('pip install numpy')
    import numpy as np

try:
    from stop_words import get_stop_words
except ModuleNotFoundError:
    os.system('pip install stop_words')
    from stop_words import get_stop_words

stop_words = get_stop_words ('spanish')
data1 = []
data2 = ['Yo quiero al perro']
data3 = ['Yo sí quiero al gato']

data1.append(data2[0])
data1.append(data3[0])
print(data1)

#jackard S, dice, levenshtein D
from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer().fit_transform(data1)

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(tfid[1],tfid[0]).flatten())