import json
import MeCab
from functools import reduce
import itertools

from collections import Counter

mecab = MeCab.Tagger('')

# 半分テスト、半分学習用
# return pair
def _create_train_and_test_words(filename):
    with open(filename) as f:
        words_s = []
        test_words_s = []
        i = 0
        for line in f.readlines():
            j = json.loads(line)
            names = []
            if "ingredients" not in j:
                continue
            l = [x["name"] for x in j["ingredients"]]
            l.append(j["title"])
            for name in l:
                # たまにエラーがでるので
                try:
                    for word in filter(lambda x: "\t名詞,一般" in x or "\t名詞,固有名詞" in x, mecab.parse(name).split("\n")):
                         names.append(word.split("\t")[0])
                except:
                    continue
            if i % 2 == 0:
                words_s.append(names)
            else:
                test_words_s.append(names)
            i += 1
        return (words_s, test_words_s)

def _create_vector_from_word(original_words, vector_words):
    vector = [0] * len(original_words)
    for word in vector_words:
        try:
            vector[original_words.index(word)] += 1
        except:
            continue
    return vector

positive_words_s, test_positive_words_s = _create_train_and_test_words("positive_recipe.json")

positive_counts = Counter(itertools.chain.from_iterable(positive_words_s)).most_common(200)

negative_words_s, test_negative_words_s = _create_train_and_test_words("negative_recipe.json")

negative_counts = Counter(itertools.chain.from_iterable(negative_words_s)).most_common(200)

words = list(set([x[0] for x in positive_counts] + [x[0] for x in negative_counts]))

# word list

test_word_vectors = []
test_word_bools = []

fit_word_vectors = []
fit_word_bools = []

for i in range(0, len(positive_words_s)):
    positive_words = positive_words_s[i]
    fit_word_vectors.append(_create_vector_from_word(words, positive_words))
    fit_word_bools.append(1)

for i in range(0, len(negative_words_s), 2):
    negative_words = negative_words_s[i]
    fit_word_vectors.append(_create_vector_from_word(words, negative_words))
    fit_word_bools.append(0)

for test_positive_words in test_positive_words_s:
    test_word_vectors.append(_create_vector_from_word(words, test_positive_words))
    test_word_bools.append(1)

for i in range(0, len(test_negative_words_s), 15):
    test_negative_words = test_negative_words_s[i]
    test_word_vectors.append(_create_vector_from_word(words, test_negative_words))
    test_word_bools.append(0)

print("test data create ok")

from sklearn import svm
clf = svm.SVC()
clf.fit(fit_word_vectors, fit_word_bools)
c = 0
for vector, boo in zip(test_word_vectors, test_word_bools):
    a = clf.predict([vector])
    if a[0] == boo:
        c += 1

print("ok count: {0}".format(c))
print("accuracy: {0}".format(c / len(test_word_vectors)))
