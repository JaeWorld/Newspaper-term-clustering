import pickle
import pandas as pd
import numpy as np
from konlp.kma.klt2000 import klt2000
import re

print("Test log 2")

class TextRank:
    def __init__(self):
        self.num_of_words = 0
        self.d = 0.85
        self.window_size = 2
        self.keyword = input("키워드를 입력해주세요: ")
        print("키워드: ", self.keyword)
    
    # 메인 실행 함수
    # input: articles (article 모은 리스트)
    def text_rank(self, articles):
        articles_tokenized = self.preprocess_articles(articles)
        n_grams = self.n_grams(articles_tokenized)
        word_index = self.word_index_n_grams(n_grams)
        matrix = self.term_matrix(n_grams, word_index)
        pagerank = self.iteration(matrix, word_index)
        self.result(pagerank, word_index)
            
    # Article내에서 각 sentence 토큰화
    # input: article (단일 article)
    def preprocess_sent(self, article):
        sentences = []
        k = klt2000()
        
        for sent in article:
            sent = re.sub('[\W0-9]', ' ', sent)
            tokenized = k.morphs(sent)
            tokenized = [word for word in tokenized if len(word) >= 2]
            sentences.append(tokenized)        
        return sentences
    
    # 모든 Articles 토큰화
    # input: articles (article 모은 리스트)
    def preprocess_articles(self, articles):
        articles_tokenized = []
        for article in articles:
            sentences = self.preprocess_sent(article)
            articles_tokenized.append(sentences)
        return articles_tokenized

    # Article의 각 단어 출현횟수 카운트 
    # input: sentences (문장 리스트)
    def word_count(self, sentences):
        word_count = {}
        for sentence in sentences:
            for word in sentence:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        return word_count
    
    # 문장 리스트를 입력으로 받아 word_index 생성
    # input: sentences (문장 리스트)
    def word_index(self, sentences):
        word_index = {}
        idx = 0
        for sentence in sentences:
            for word in sentence:
                if word not in word_index:
                    word_index[word] = idx
                    idx += 1
                    
        self.num_of_words = len(word_index)
        return word_index
    
    # 단어 리스트를 입력으로 받아 word_index 생성
    # input: words (단어 리스트)
    def word_index_word_list(self, words):
        word_index = {}
        idx = 0
        for word in words:
            if word not in word_index:
                word_index[word] = idx
                idx += 1
        return word_index
    
    # keyword근처에 있는 단어들로 n_gram 생성
    # input: articles_tokenized (articles의 모든 단어들 토큰화한 리스트)
    def n_grams(self, articles_tokenized):
        n_grams = []
        for article in articles_tokenized:
            for sentence in article:
                for i in range(len(sentence)):
                    if sentence[i] == self.keyword:
                        n_gram = []
                        for j in range(i-self.window_size, i+self.window_size):
                            if j <= 0 or j >= len(sentence):
                                continue
                            n_gram.append(sentence[j])
                        n_grams.append(n_gram)
        return n_grams
        
    # window_size에 맞춰 word pair 생성
    # input: sentences (문장 리스트)
    def pair_words(self, sentences):
        pairs = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+self.window_size):
                    if j >= len(sentence):
                        break
                    pairs.append([sentence[i], sentence[j]])
        return pairs
    
    # n_grams을 입력으로 받아 word_index 생성
    # input: n_grams
    def word_index_n_grams(self, n_grams):
        words = set()
        for n_gram in n_grams:
            for word in n_gram:
                words.add(word)
        word_index = self.word_index_word_list(words)
        return word_index
    
    # word pairs z입력받아 term to term matrix 생성
    # input: n_grams, word_index
    def term_matrix(self, n_grams, word_index):
        matrix = np.zeros((len(word_index), len(word_index)))

        for n_gram in n_grams:
            for w1 in n_gram:
                for w2 in n_gram:
                    if w1 == w2:
                        continue
                    w1_idx, w2_idx = word_index[w1], word_index[w2]
                    matrix[w1_idx][w2_idx] += 1
                    matrix[w2_idx][w1_idx] += 1
            
        sum_of_matrix = np.sum(matrix)
        matrix = np.divide(matrix, sum_of_matrix)
        return matrix
    
    # pagerank 알고리즘 반복
    # input: matrix, word_index
    def iteration(self, matrix, word_index):
        pagerank = np.array([1]*len(word_index))

        for epoch in range(10):
            pagerank = (1-self.d) + self.d*np.dot(matrix, pagerank)
        return pagerank
    
    # 출력 결과
    # input: pagerank (페이지랭크 점수), word_index
    def result(self, pagerank, word_index):
        node_weight = {}
        for word, index in word_index.items():
            node_weight[word] = pagerank[index]

        res = sorted(node_weight.items(), key=lambda x: -x[1])
        print("=== 상위 10개 키워드 ===")
        print(print([[word[0], round(word[1], 3)] for word in res[:10]]), '\n')
