from flask import request, jsonify
from app import app
from app.model import commission_station_process


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    file_list = data.get('dir')
    thresh = 0.01  # request.args.get('thresh')

    if not isinstance(file_list, list):
        return jsonify({"error": "요청 포맷이 맞지 않음"}), 400

    # 비동기 처리 시작
    commission_station_process(file_list, thresh)

    # 200 즉시 응답
    return jsonify({"message": "요청 처리 중"}), 200
