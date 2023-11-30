from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import math

import pika
import json

# Word2Vec 모델 불러오기
model = Word2Vec.load('W2V모델')
vectorized_df=pd.read_csv('벡터화데이터.csv')

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

def callback(ch, method, properties, body):
    # Received message
    global model, vectorized_df
    message = body.decode('utf-8')
    print(f" [*] Received message: {message}")
    data = json.loads(message)
    print(f'data : {data}')
    
    # 우리가 짠 코드 시작
    selected_titles = eval(message)  # 선택한 영상의 제목
    if len(selected_titles) == 0:
        similar_indices = np.random.randint(0, vectorized_df.shape[0], size=60)
    else:
        selected_titles_df = pd.DataFrame(data=selected_titles, columns=['제목'])
        selected_titles_df['제목'] = selected_titles_df['제목'].str.lower()
        selected_titles_df['제목'] = selected_titles_df['제목'].str.replace('[^\w\s]', ' ', regex=True)
        selected_titles_df['제목'] = selected_titles_df['제목'].str.replace('\s+', ' ', regex=True)
        selected_titles_df['제목'] = selected_titles_df['제목'].str.strip()
        filtered_vectorized_df=vectorized_df.iloc[:,1:]
        #모든 영상 벡터
        all_vectors = filtered_vectorized_df.values
        selected_titles_df['제목토큰'] = selected_titles_df['제목'].apply(word_tokenize)
        selected_vectors = pd.DataFrame(selected_titles_df['제목토큰'].apply(lambda x: get_sentence_vector(x,model)).tolist())
        similarity_scores = cosine_similarity(selected_vectors, all_vectors)
        num_sim = math.ceil(200/len(selected_titles))+1
        similar_indices = []
        for index in range(len(selected_titles)):
            similar_indices.extend(similarity_scores[index].argsort()[:-num_sim:-1])

        # 중복 제거 및 상위 24개 영상 출력
        pd.set_option('display.max_colwidth', None)
        for title in selected_titles:
            print('선택한 영상: ' + title)
        print()
        np.random.shuffle(similar_indices)
        similar_indices = list(dict.fromkeys(similar_indices))

    recommended_videos = vectorized_df['영상 ID'].iloc[similar_indices[:60]].tolist()
    # print(recommended_videos['영상 ID'])
    #우리가 짠 코드 끝

    # Process the message (you can add your logic here)
    response_message = 'Reply python'

    # Send the response back
    ch.basic_publish(
        exchange='',
        routing_key=properties.reply_to,
        properties=pika.BasicProperties(
            correlation_id=properties.correlation_id,
        ),
        body=str(recommended_videos),
    )

    print(f" [x] Sent response: {recommended_videos}")

# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare a queue
queue_name = 'my_queue'
channel.queue_declare(queue=queue_name)

# Set up the callback function to handle incoming messages
channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

print(' [*] Waiting for messages. To exit, press CTRL+C')
channel.start_consuming()


# 도커 열어서 rabbitMQ 실행 뒤 실행
# docker run -d --name rabbitmq -p 5672:5672 -p 8080:15672 --restart=unless-stopped rabbitmq:management
# docker exec -it rabbitmq bash 
