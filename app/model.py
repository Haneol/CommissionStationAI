import json
import os
import threading
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
import re

from app.tokenizer import kiwi_tokenizer
from app.boto import read_file_from_s3, parse_s3_link, read_text_file
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer


def predict_process(file, thresh, count_vectorizer, tfidf_transformer, tflite_model):
    # bucket, key = parse_s3_link(file)
    # data = read_file_from_s3(bucket, key)
    data = read_text_file(file)

    data_len, data = text_preprocesssing(data)
    if len(data) == 0:
        return False

    data = [word for word in data if any(char.isspace() for char in word)]

    data = tf_idf_vectorize(data, count_vectorizer, tfidf_transformer)
    pred = model_predict(data_len, data, tflite_model)
    print(pred)
    if pred >= thresh:
        return True
    else:
        return False


def text_preprocesssing(data):
    filter_words = ['커미션', 'commission', '일러', '그림', '그리', '그린', '그릴', '그려', '그렸', '판매', '팔아' '판', '작업', '외주', '후원',
                    '신청']
    soup = BeautifulSoup(data, 'html.parser')

    # <head> 제거
    if soup.head:
        soup.head.decompose()

    # <style> 제거
    for style in soup.find_all('style'):
        style.decompose()

    # <script> 제거
    for script in soup.find_all('script'):
        script.decompose()

    # Text만 추출
    text = soup.get_text(separator='\n', strip=True)

    # URL, 이메일, 전화번호 제거
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'(\d{2,3}[-.\s]??\d{3,4}[-.\s]??\d{4})|(\d{10,11})', '', text)

    # 특수문자 제거
    text = re.sub(r'[^\w\s.?ㄱ-ㅎ가-힣]+', ' ', text)

    # 소문자로 변경
    text = text.lower()

    # 연속되는 공백을 하나의 공백으로 압축
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    text = text.split('\n')
    text = list(set(text))

    # filter_words에 있는 단어를 포함하고 빈 문자열이 아닌 요소만 리스트에 포함
    filtered_text = [sent for sent in text if any(fw in sent for fw in filter_words) and sent.strip()]

    return len(text), filtered_text


def tf_idf_vectorize(data, count_vectorizer, tfidf_transformer):
    count_vectors = count_vectorizer.transform(data)
    tfidf_vectors = tfidf_transformer.transform(count_vectors).toarray()

    return tfidf_vectors


def model_predict(data_len, data, tflite_model_file):
    thresh = 0.5
    interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 확률 평균 계산용 변수
    num_pred = data_len
    sum_pred = 0

    for item in data:
        item = np.expand_dims(item, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], item)

        # 인터프리터 실행
        interpreter.invoke()

        # 출력 텐서에서 결과 얻기
        output = interpreter.get_tensor(output_details[0]['index'])

        # 결과 저장
        if output >= thresh:
            sum_pred += 1

    return sum_pred / num_pred


def predict_process_all(file_list, thresh):
    base_path = os.path.abspath(os.path.dirname(__file__))

    count_vectorizer_path = os.path.join(base_path, '..', 'models', 'count_vectorizer_vocabulary.joblib')
    tfidf_transformer_path = os.path.join(base_path, '..', 'models', 'tfidf_transformer.joblib')
    tflite_model_path = os.path.join(base_path, '..', 'models', 'commission_station_model_quantized_pruning.tflite')

    vocabulary = load(count_vectorizer_path)
    count_vectorizer = CountVectorizer(tokenizer=kiwi_tokenizer, vocabulary=vocabulary)
    tfidf_transformer = load(tfidf_transformer_path)

    results = {}
    for file in file_list:

        print(f'predicting... {file}')
        result = predict_process(file, thresh, count_vectorizer, tfidf_transformer, tflite_model_path)
        results[file] = result

    result_json = json.dumps(results, indent=4)
    print(f'[DATA]\n{result_json}\n\n')

    # response = requests.post("http://localhost:8080/", data=result_json)
    # print(f'[RESPONSE] Response status code: {response.status_code}')
    # print(response)


def commission_station_process(file_list, thresh):
    threading.Thread(target=predict_process_all, args=(file_list, thresh,)).start()
