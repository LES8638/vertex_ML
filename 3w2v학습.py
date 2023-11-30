from gensim.models import Word2Vec
import pandas as pd
df = pd.read_csv('학습데이터.csv', converters={'제목': eval})
sentences=df['제목'].tolist()
model = Word2Vec(sentences=sentences, vector_size=200, window=5, min_count=2, workers=4)
model.save("W2V모델")