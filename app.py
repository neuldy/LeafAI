from flask import Flask, render_template, request, redirect, url_for, Response
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import time
import threading
from picamera2 import Picamera2
from werkzeug.utils import secure_filename  # Import secure_filename

# Flask 설정
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 업로드 폴더 생성
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 모델 및 클래스 매핑 로드
MODEL_PATH = 'Leafy.h5'
CLASS_MAPPING_PATH = 'Leafy_class_mapping.json'

# 모델 로드
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# 클래스 매핑 로드
if not os.path.exists(CLASS_MAPPING_PATH):
    raise FileNotFoundError(f"Class mapping file not found: {CLASS_MAPPING_PATH}")
with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
    CLASS_NAMES = json.load(f)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Picamera2 전역 객체 및 락 생성
picam2 = None
camera_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_camera():
    global picam2
    with camera_lock:
        if picam2 is None:
            picam2 = Picamera2()
            picam2.configure(picam2.create_still_configuration())
            picam2.start()

@app.route('/')
def home():
    return render_template('home.html')

def extract_keyword(predicted_class_name):
    keyword_map = {
        "corn": "corn",
        "apple": "apple",
        "orange": "orange",
        "tomato": "tomato",
        "bean": "bean",
        "bellpepper": "bellpepper",
        "blueberry": "blueberry",
        "ginger": "ginger",
        "grapes": "grapes",
        "lemon": "lemon",
        "onion": "onion",
        "peach": "peach",
        "potato": "potato",
        "raspberry": "raspberry",
        "rice": "rice",
        "strawberry": "strawberry",
    }
    for keyword, filename in keyword_map.items():
        if keyword in predicted_class_name.lower():
            return filename  # 항상 소문자로 반환
    return "unknown"  # unknown.html이 있어야 함

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method == 'POST':
        try:
            file = request.files['file']
            if file and allowed_file(file.filename):
                # 파일 저장
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # 'uploads' 디렉토리 경로 제거
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)

                # 모델 예측 및 아두이노로 전송
                result, confidence = predict_and_send(filepath)

                # 신뢰도가 낮아 No Data인 경우 unknown.html로 이동
                if result == "No Data":
                    return redirect(url_for('plant_result', crop='unknown'))

                # URL 생성 및 작물 키워드 추출
                image_url = url_for('static', filename=f'uploads/{filename}')
                crop = extract_keyword(result)

                # 결과 리디렉션
                return redirect(
                    url_for('plant_result', crop=crop, filepath=image_url, result=result, confidence=f"{confidence:.2f}")
                )
            else:
                return "Invalid file format. Allowed formats: png, jpg, jpeg, bmp", 400
        except Exception as e:
            return f"Error during upload: {str(e)}", 500


@app.route('/capture', methods=['GET', 'POST'])
def capture():
    initialize_camera()  # Picamera2 초기화
    if request.method == 'GET':
        return render_template('capture.html')
    elif request.method == 'POST':
        try:
            # 파일 저장 경로 설정
            timestamp = int(time.time())
            filename = f"captured_{timestamp}.jpg"
            save_dir = os.path.join("static", "captures")
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)

            # 이미지 캡처 및 저장
            with camera_lock:
                picam2.capture_file(filepath)

            print(f"[INFO] Image successfully saved to: {filepath}")

            # 예측 실행 및 키워드 추출
            result, confidence = predict_and_send(filepath)

            # 신뢰도가 낮아 No Data인 경우 unknown.html로 이동
            if result == "No Data":
                return redirect(url_for('plant_result', crop='unknown'))

            crop = extract_keyword(result)

            # 결과 페이지로 리디렉션
            return redirect(
                url_for('plant_result', crop=crop, filepath=url_for('static', filename=f'captures/{filename}'), result=result, confidence=f"{confidence:.2f}")
            )
        except Exception as e:
            print(f"[ERROR] Error during image capture: {str(e)}")
            return f"Error during image capture: {str(e)}"

@app.route('/plant_result/<crop>')
def plant_result(crop):
    try:
        # crop 값을 소문자로 변환
        crop_lower = crop.lower()
        filepath = request.args.get('filepath')  # 이미지 경로 (선택적)
        result = request.args.get('result')  # 예측 결과 (선택적)
        confidence = request.args.get('confidence')  # 신뢰도 (선택적)

        # 디버깅 메시지 출력
        print(f"[DEBUG] crop_lower: {crop_lower}")
        print(f"[DEBUG] filepath: {filepath}")
        print(f"[DEBUG] result: {result}")
        print(f"[DEBUG] confidence: {confidence}")

        # unknown.html로 리디렉션
        if crop_lower == "unknown":
            return render_template("plant_result/unknown.html")

        # 이미지 경로와 결과가 없는 경우(검색 요청)
        if not filepath and not result and not confidence:
            # 단순히 crop에 대한 HTML 페이지 렌더링
            return render_template(f"plant_result/{crop_lower}.html")

        # 이미지 경로와 결과가 있는 경우(예측 결과 페이지)
        return render_template(
            f"plant_result/{crop_lower}.html",
            filepath=filepath,
            result=result,
            confidence=confidence
        )
    except Exception as e:
        print(f"[ERROR] Unable to load template for crop: {crop}. Details: {str(e)}")
        return f"Error: Unable to load the page for {crop}. Details: {str(e)}", 404

def generate_frames():
    initialize_camera()  # Picamera2 초기화
    while True:
        with camera_lock:  # 락 사용
            frame = picam2.capture_array()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cctv')
def cctv():
    return render_template('cctv.html')

import atexit

@atexit.register
def cleanup_camera():
    global picam2
    with camera_lock:
        if picam2:
            picam2.stop()
            print("[INFO] Camera stopped successfully.")

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        plant_name = request.form.get('plant_name', '').strip().lower()  # 입력값 가져오기
        if plant_name:  # 입력값이 있는 경우
            return redirect(url_for('plant_result', crop=plant_name))  # crop에 매핑
        else:
            return "Please enter a valid plant name.", 400
    return render_template('search.html')  # 검색 페이지 렌더

import serial  # 아두이노 통신을 위한 라이브러리
import time  # 통신 대기용

# 기존 모델 예측 함수 수정
from send_to_arduino import send_to_arduino

def predict(image_path):
    try:
        # 이미지를 로드 및 전처리
        image = load_img(image_path, target_size=(128, 128))  # 모델 입력 크기에 맞게 크기 변경
        image_array = img_to_array(image) / 255.0  # 정규화
        image_array = np.expand_dims(image_array, axis=0)  # 배치 차원 추가

        # 모델 예측
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])  # 가장 높은 확률의 클래스
        confidence = predictions[0][predicted_class_index]
        predicted_class_name = CLASS_NAMES[str(predicted_class_index)]

        return predicted_class_name, confidence
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

def predict_and_send(image_path):
    try:
        # 모델 예측 수행
        predicted_class_name, confidence = predict(image_path)
        print(f"[DEBUG] Predicted class: {predicted_class_name}, Confidence: {confidence:.2f}")

        # 신뢰도가 낮아 No Data로 처리
        if confidence < 0.25:
            print("[DEBUG] Confidence below threshold. Treating as No Data.")
            send_to_arduino('Y')  # 노란불
            return "No Data", confidence

        # 결과에 따라 Arduino로 데이터 전송
        if "Healthy" in predicted_class_name:
            print("[DEBUG] Sending 'G' to Arduino for green LED (Healthy)")
            send_to_arduino('G')  # 초록불
        else:
            print("[DEBUG] Sending 'R' to Arduino for red LED (Disease detected)")
            send_to_arduino('R')  # 빨간불

        return predicted_class_name, confidence
    except Exception as e:
        print(f"[ERROR] Prediction and Arduino communication failed: {e}")
        send_to_arduino('O')  # 모든 LED 끄기
        return "No Data", 0.0

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
