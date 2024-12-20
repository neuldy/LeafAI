import serial
import time

def send_to_arduino(command):
    """
    아두이노로 명령을 전송합니다.

    Args:
        command (str): 아두이노로 보낼 명령. 'R', 'G', 'Y', 'O' 중 하나.
    """
    try:
        # 시리얼 포트 설정 (아두이노가 연결된 포트 확인)
        arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        time.sleep(2)  # 아두이노 초기화 대기

        # 명령 전송
        arduino.write(command.encode())
        print(f"[INFO] Sent to Arduino: {command}")

        # 연결 종료
        arduino.close()
    except Exception as e:
        print(f"[ERROR] Failed to send to Arduino: {e}")
