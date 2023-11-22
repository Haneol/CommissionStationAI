from flask import request, jsonify
from app import app
from app.model import commission_station_process


@app.route('/predict')
def predict():
    directory = request.args.get('route')
    thresh = 0.15  # request.args.get('thresh')

    if not directory:
        return jsonify({"message": "경로 설정 필요"}), 400

    # 비동기 처리 시작
    commission_station_process(directory, thresh)

    # 200 즉시 응답
    return jsonify({"message": "요청 처리 중"}), 200
