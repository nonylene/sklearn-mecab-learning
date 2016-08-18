import json
import MeCab
from functools import reduce

# TODO: refactoring

mecab = MeCab.Tagger('')
with open("positive_recipe.json") as f:
    # 材料
    positive_words_s = []
    for line in f.readlines():
        j = json.loads(line)
        names = []
        if "ingredients" not in j:
            continue
        for name in list( map(lambda x: x["name"], j["ingredients"])):
            # なぜかエラーがでるので
            try:
                for word in filter(lambda x: "\t名詞," in x, mecab.parse(name).split("\n")):
                     names.append(word.split("\t")[0])
            except:
                continue
        positive_words_s.append(names)

# 頻度が少ないものを取り除く

positive_counts = {}

for positive_words in positive_words_s:
    for word in positive_words:
        if word in positive_counts:
            positive_counts[word] += 1
        else:
            positive_counts[word] = 1

for c in list(positive_counts):
    if positive_counts[c] < 30:
        del positive_counts[c]


with open("negative_recipe.json") as f:
    # 材料
    negative_words_s = []
    for line in f.readlines():
        j = json.loads(line)
        names = []
        if "ingredients" not in j:
            continue
        for name in list( map(lambda x: x["name"], j["ingredients"])):
            # なぜかエラーがでるので
            try:
                for word in filter(lambda x: "\t名詞," in x, mecab.parse(name).split("\n")):
                     names.append(word.split("\t")[0])
            except:
                continue
        negative_words_s.append(names)

# 頻度が少ないものを取り除く

negative_counts = {}

for negative_words in negative_words_s:
    for word in negative_words:
        if word in negative_counts:
            negative_counts[word] += 1
        else:
            negative_counts[word] = 1

for c in list(negative_counts):
    if negative_counts[c] < 700:
        del negative_counts[c]

words = list(set(list(positive_counts) + list(negative_counts)))

print(len(words))

# word list

word_vectors = []
word_bools = []
fit_word_vectors = []
fit_word_bools = []

for i in range(len(positive_words_s)):
    positive_words = positive_words_s[i]
    vector = [0] * len(words)
    for word in positive_words:
        if word in words:
            vector[words.index(word)] += 1
    if i % 2 == 0:
        fit_word_vectors.append(vector)
        fit_word_bools.append(1)
    word_vectors.append(vector)
    word_bools.append(1)

for i in range(len(negative_words_s)):
    negative_words = negative_words_s[i]
    vector = [0] * len(words)
    for word in negative_words:
        if word in words:
            vector[words.index(word)] += 1
    if i % 7 == 0:
        fit_word_vectors.append(vector)
        fit_word_bools.append(0)
    if i % 15 == 0:
        word_vectors.append(vector)
        word_bools.append(0)

print("ok")

from sklearn import svm
clf = svm.SVC()
clf.fit(fit_word_vectors, fit_word_bools)
c = 0
for vector, boo in zip(word_vectors, word_bools):
    a = clf.predict([vector])
    if a[0] == boo:
        c += 1

print(c)
print(c / len(word_vectors))

