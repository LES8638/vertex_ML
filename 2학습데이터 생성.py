import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# csv 파일 읽기
df = pd.read_csv('결측치제거데이터.csv') 

#토큰화 및 불용어 처리
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

tokenized_sentences=[]

for index, row in df.iterrows():
    sentence = str(row['제목'])

    #문장 토큰화
    tokens=word_tokenize(sentence)
    
    #불용어 제거
    filtered_tokens=[token for token in tokens if token not in stop_words and re.match("^[A-Za-z0-9]*$",token) and not re.match("^\d+$", token)]

    #결과 저장
    tokenized_sentences.append(filtered_tokens)

df['제목'] = tokenized_sentences

# 결과를 새로운 csv 파일로 저장
df.to_csv('학습데이터.csv', index=False)