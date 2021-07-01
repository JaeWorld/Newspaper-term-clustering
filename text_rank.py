import pickle
import pandas as pd
import numpy as np
from konlp.kma.klt2000 import klt2000
import re

class TextRank:
    def __init__(self):
        self.num_of_words = 0
        self.d = 0.85
        self.window_size = 2
    
    # 메인 실행 파일
    def text_rank(self, article):
        sentences = self.preprocess_sent(article)
        word_index = self.word_index(sentences)
        pairs = self.pair_words(sentences)
        matrix = self.term_matrix(pairs, word_index)
        pagerank = self.iteration(matrix)
        self.result(pagerank, word_index)
        print('Article: ', article)
            
    # Article내에서 각 sentence 토큰화
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
    def preprocess_articles(self, articles):
        articles_tokenized = []
        for article in articles:
            sentences = self.preprocess_sent(article)
            articles_tokenized.append(sentences)
        return articles_tokenized

    # Article의 각 단어 출현횟수 카운트    
    def word_count(self, sentences):
        word_count = {}
        for sentence in sentences:
            for word in sentence:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        return word_count
    
    # 각 단어에 index 부여
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
    
    # window_size에 맞춰 word pair 생성
    def pair_words(self, sentences):
        pairs = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+self.window_size):
                    if j >= len(sentence):
                        break
                    pairs.append([sentence[i], sentence[j]])
        return pairs
    
    # word pairs z입력받아 term to term matrix 생성
    def term_matrix(self, pairs, word_index):
        matrix = np.zeros((len(word_index), len(word_index)))

        for pair in pairs:
            w1, w2 = pair
            w1_idx, w2_idx = word_index[w1], word_index[w2]
            matrix[w1_idx][w2_idx] += 1
            matrix[w2_idx][w1_idx] += 1
            
        sum_of_matrix = np.sum(matrix)
        matrix = np.divide(matrix, sum_of_matrix)
        return matrix
    
    # pagerank 알고리즘 반복
    def iteration(self, matrix):
        pagerank = np.array([1]*self.num_of_words)

        for epoch in range(10):
            pagerank = (1-self.d) + self.d*np.dot(matrix, pagerank)
        return pagerank
    
    # 결과 출력
    def result(self, pagerank, word_index):
        node_weight = {}
        for word, index in word_index.items():
            node_weight[word] = pagerank[index]

        res = sorted(node_weight.items(), key=lambda x: -x[1])
        print("=== 상위 5개 키워드 ===")
        print(print([word[0] for word in res[:5]]), '\n')
