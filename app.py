from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
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
    if request.method == 'POST':
        try:
            # 이미지 캡처 및 저장
            timestamp = int(time.time())
            filename = f"captured_{timestamp}.jpg"
            save_dir = os.path.join("static", "captures")
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)

            with camera_lock:
                picam2.capture_file(filepath)

            # 예측 및 결과 처리
            result, confidence = predict_and_send(filepath)

            if confidence < 0.25:
                return redirect(url_for('capture_result', result="No Data", filepath=filepath))

            crop = extract_keyword(result)

            # 결과 페이지로 리디렉션
            return redirect(
                url_for('capture_result', result=result, filepath=url_for('static', filename=f'captures/{filename}'), confidence=f"{confidence:.2f}")
            )

        except Exception as e:
            return f"Error during image capture: {str(e)}", 500
    return render_template('capture.html')  # GET 요청일 경우 capture 페이지를 반환


@app.route('/capture_result')
def capture_result():
    result = request.args.get('result')
    filepath = request.args.get('filepath')
    confidence = float(request.args.get('confidence'))

    # If confidence is below threshold (e.g., 0.4 or 40%), show "No Data" page
    if confidence < 0.4:
        return render_template('capture_result.html', result="No Data", filepath=filepath, confidence=None)

    # If confidence is above threshold, redirect to specific crop page
    crop = extract_keyword(result)  # Extract the crop keyword (e.g., 'cabbage', 'tomato')
    
    # Ensure the filepath and result are passed correctly
    return redirect(url_for('plant_result', crop=crop, filepath=filepath, result=result, confidence=f"{confidence:.2f}"))


@app.route('/plant_result/<crop>')
def plant_result(crop):
    try:
        crop_lower = crop.lower()  # Ensure crop name is in lowercase
        filepath = request.args.get('filepath')  # Optional image path
        result = request.args.get('result')  # Optional prediction result
        confidence = request.args.get('confidence')  # Optional confidence value

        print(f"[DEBUG] crop_lower: {crop_lower}")
        print(f"[DEBUG] filepath: {filepath}")
        print(f"[DEBUG] result: {result}")
        print(f"[DEBUG] confidence: {confidence}")

        # If no result, treat as the crop name
        if not result:
            result = crop  # Set result to crop name if no prediction is available

        # Default image handling
        if not filepath:
            filepath = url_for('static', filename=f'img/{crop_lower}/{crop_lower}.jpg')

        # Ensure confidence is a float
        confidence_value = float(confidence) if confidence else 0.0
        confidence_percentage = round(confidence_value * 100, 2)

        # If 'No Data' result is detected, render a page showing only crop name
        if result == "No Data":
            return render_template("plant_result/unknown.html", crop=crop)

        return render_template(
            f"plant_result/{crop_lower}.html",
            filepath=filepath,
            result=result,
            confidence=confidence_percentage
        )

    except Exception as e:
        print(f"[ERROR] Unable to load template for crop: {crop}. Details: {str(e)}")
        return f"Error: Unable to load the page for {crop}. Details: {str(e)}", 404

import torch
import cv2

def generate_frames():
    initialize_camera()  # Initialize Picamera2

    # Load YOLOv5 model (default model can be 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

    while True:
        with camera_lock:  # Use lock for thread safety
            frame = picam2.capture_array()  # Capture frame from Pi Camera

            # Perform object detection on the captured frame using YOLOv5
            results = model(frame)  # Perform inference
            print("[DEBUG] Detection Results:", results.pandas().xywh)  # Show detected objects info

            # Render the results on the frame
            frame = results.render()[0]  # Render the detection results on the frame

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame for streaming
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
    
arduino_port = '/dev/ttyUSB0'  # Update this to match your Arduino port
baud_rate = 9600
try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)  # Allow Arduino to reset
except Exception as e:
    print(f"[ERROR] Unable to connect to Arduino: {e}")
    arduino = None

def predict_and_send(image_path):
    try:
        # 모델 예측 수행
        predicted_class_name, confidence = predict(image_path)
        print(f"[DEBUG] Predicted class: {predicted_class_name}, Confidence: {confidence:.2f}")

        # 신뢰도가 낮아 No Data로 처리
        if confidence < 0.4:  # 40% 이하일 때 No Data 처리
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



@app.route('/get_sensor_data', methods=['POST'])
def get_sensor_data():
    try:
        if arduino:
            arduino.write(b'S')  # Send 'S' to Arduino to request sensor data
            time.sleep(1)
            data = arduino.readline().decode('utf-8').strip()
            if data:
                return jsonify({"status": "success", "data": data})
            else:
                return jsonify({"status": "error", "message": "No data received from Arduino."})
        else:
            return jsonify({"status": "error", "message": "Arduino not connected."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/sensor_control', methods=['GET'])
def sensor_control():
    try:
        if arduino:
            arduino.write(b'S')  # Arduino로 센서 데이터 요청
            time.sleep(1)
            data = arduino.readline().decode('utf-8').strip()  # Arduino에서 데이터 수신
            
            print(f"[DEBUG] Raw data from Arduino: {data}")  # 디버깅용 출력

            if data:
                try:
                    # 데이터 파싱
                    # 예: "Temperature: 1212.40 °C, PPFD: 170.46 μmol/m²/s"
                    parts = data.split(", ")
                    temperature_part = parts[0].split(":")[1].strip().replace("°C", "").strip()
                    ppfd_part = parts[1].split(":")[1].strip().replace("μmol/m²/s", "").strip()

                    # 온도와 PPFD 값을 float으로 변환
                    temperature = float(temperature_part)
                    ppfd = float(ppfd_part)

                    return render_template(
                        'sensor_control.html',
                        temperature=f"{temperature:.2f}",
                        ppfd=f"{ppfd:.2f}"
                    )
                except (IndexError, ValueError) as parse_error:
                    print(f"[ERROR] Data parsing failed: {parse_error}")
                    return render_template(
                        'sensor_control.html',
                        temperature="Data Parsing Error",
                        ppfd="Data Parsing Error"
                    )
            else:
                return render_template('sensor_control.html', temperature="No Data", ppfd="No Data")
        else:
            return render_template('sensor_control.html', temperature="Arduino Not Connected", ppfd="N/A")
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        return render_template('sensor_control.html', temperature="Error", ppfd=str(e))


GROWTH_LOG_FILE = 'growth_logs.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
from datetime import datetime

@app.route('/growth_log', methods=['GET', 'POST'])
def growth_log():
    try:
        if arduino:
            arduino.write(b'S')  # Send 'S' to Arduino to request sensor data
            time.sleep(1)
            data = arduino.readline().decode('utf-8').strip()
            if data:
                # Parse Arduino data, e.g., "Temperature: 25.50 °C, PPFD: 150.30 μmol/m²/s"
                parts = data.split(", ")
                temperature_part = parts[0].split(":")[1].strip().replace("°C", "").strip()
                ppfd_part = parts[1].split(":")[1].strip().replace("μmol/m²/s", "").strip()
                temperature = float(temperature_part)
                ppfd = float(ppfd_part)
            else:
                temperature, ppfd = None, None
        else:
            temperature, ppfd = None, None

    except Exception as e:
        print(f"[ERROR] Failed to read from Arduino: {e}")
        temperature, ppfd = None, None

    if request.method == 'POST':
        crop_name = request.form['crop_name']
        # Remove the date handling here as we no longer need it
        # We don't use date from the form anymore, as requested
        
        watering = request.form.get('water', 'unknown')  # 기본값 설정
        humidity = request.form.get('humidity', 'unknown')  # 기본값 설정
        weather = request.form.get('weather', 'unknown')  # 기본값 설정
        notes = request.form['notes']

        # Save the log entry with temperature and PPFD
        growth_log_entry = {
            'crop_name': crop_name,
            'watering': watering,
            'weather': weather,
            'notes': notes,
            'temperature': temperature,
            'ppfd': ppfd,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'humidity': humidity
        }

        # Read or initialize growth logs
        if os.path.exists(GROWTH_LOG_FILE):
            with open(GROWTH_LOG_FILE, 'r', encoding='utf-8') as f:
                growth_logs = json.load(f)
        else:
            growth_logs = []

        growth_logs.append(growth_log_entry)

        # Save updated logs
        with open(GROWTH_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(growth_logs, f, ensure_ascii=False, indent=4)

        return redirect(url_for('view_growth_logs'))

    # Removed the logic for passing today's date to the template, since it's not needed
    return render_template('growth_log.html', temperature=temperature, ppfd=ppfd)

@app.route('/view_growth_logs', methods=['GET'])
def view_growth_logs():
    search_query = request.args.get('q', '').strip()  # 검색어 가져오기
    growth_logs = []

    # JSON 파일에서 데이터 읽기
    if os.path.exists(GROWTH_LOG_FILE):
        with open(GROWTH_LOG_FILE, 'r', encoding='utf-8') as f:
            growth_logs = json.load(f)

    # 검색어가 있을 경우 필터링
    if search_query:
        filtered_logs = [
            log for log in growth_logs
            if search_query.lower() in log['crop_name'].lower() or
               search_query.lower() in log['notes'].lower()
        ]
    else:
        filtered_logs = growth_logs  # 검색어가 없으면 전체 데이터 사용

    return render_template('view_growth_logs.html', growth_logs=filtered_logs, search_query=search_query)



@app.route('/get_temperature', methods=['GET'])
def get_temperature():
    try:
        if arduino:
            arduino.write(b'S')  # Arduino로 데이터 요청
            time.sleep(1)
            data = arduino.readline().decode('utf-8').strip()  # Arduino에서 데이터 수신
            print(f"[DEBUG] Raw data from Arduino: {data}")  # 디버깅용 출력

            if data:
                try:
                    # 데이터 파싱
                    # 예: "Temperature: 25.50 °C, PPFD: 150.30 μmol/m²/s"
                    parts = data.split(", ")
                    temperature_part = parts[0].split(":")[1].strip().replace("°C", "").strip()
                    ppfd_part = parts[1].split(":")[1].strip().replace("μmol/m²/s", "").strip()

                    # 온도와 조도 값을 float으로 변환
                    temperature = float(temperature_part)
                    ppfd = float(ppfd_part)

                    # JSON 형태로 반환
                    return jsonify({"temperature": f"{temperature:.2f} °C", "ppfd": f"{ppfd:.2f} μmol/m²/s"})
                except (IndexError, ValueError) as parse_error:
                    print(f"[ERROR] Data parsing failed: {parse_error}")
                    return jsonify({"error": "Data parsing failed", "message": str(parse_error)}), 400
            else:
                return jsonify({"error": "No data received from Arduino"}), 400
        else:
            return jsonify({"error": "Arduino not connected"}), 500
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        return jsonify({"error": "An error occurred", "message": str(e)}), 500

@app.route('/delete_growth_log/<int:log_id>', methods=['POST'])
def delete_growth_log(log_id):
    if os.path.exists(GROWTH_LOG_FILE):
        with open(GROWTH_LOG_FILE, 'r', encoding='utf-8') as f:
            growth_logs = json.load(f)
    else:
        growth_logs = []

    # 삭제할 로그를 제외한 나머지 로그만 저장
    growth_logs = [log for idx, log in enumerate(growth_logs) if idx != log_id]

    # 업데이트된 로그를 저장
    with open(GROWTH_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(growth_logs, f, ensure_ascii=False, indent=4)

    return redirect(url_for('view_growth_logs'))

from markupsafe import Markup

@app.template_filter('highlight')
def highlight(text, search_query):
    if not search_query:
        return text
    highlighted = text.replace(
        search_query, f"<span class='highlight'>{search_query}</span>"
    )
    return Markup(highlighted)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
