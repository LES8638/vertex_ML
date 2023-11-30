import pandas as pd

df = pd.read_csv('원본데이터.csv')

df = df[df['영상 길이 (초)'] != 0]

# '제목'과 '해시태그' 컬럼의 대문자를 소문자로 바꾸기
df['제목'] = df['제목'].str.lower()

# '제목' 컬럼의 특수문자를 공백으로 대체
df['제목'] = df['제목'].str.replace('[^\w\s]', ' ', regex=True)

# 연속된 여러 개의 공백을 하나의 공백으로 대체
df['제목'] = df['제목'].str.replace('\s+', ' ', regex=True)

#처음과 끝에 있는 공백 제거
df['제목'] = df['제목'].str.strip()

#중복 제거
df=df.drop_duplicates()

df.to_csv('결측치제거데이터.csv', index=False)