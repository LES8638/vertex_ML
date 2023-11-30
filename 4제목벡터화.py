import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


# 제목을 대표하는 벡터를 계산하는 함수
def get_sentence_vector(sentence,model):
    vector = np.zeros(model.vector_size)
    count = 0
    for word in sentence:
        if word in model.wv:
            vector += model.wv[word]
            count += 1
    if count > 0:
        vector /= count
    return vector

# Word2Vec 모델 불러오기
model = Word2Vec.load('W2V모델')

# CSV 파일 불러오기
df = pd.read_csv('결측치제거데이터.csv')
df['제목'] = df['제목'].astype(str)

df['제목토큰'] = df['제목'].apply(word_tokenize)

vectors = df['제목토큰'].apply(lambda x: get_sentence_vector(x,model))

# 벡터를 데이터프레임으로 변환
vector_df = pd.DataFrame(vectors.tolist())
df=pd.concat([df['영상 ID'], vector_df], axis=1)
df.to_csv('벡터화데이터.csv', index=False)