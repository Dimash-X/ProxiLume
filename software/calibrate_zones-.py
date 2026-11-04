# ============================================================
# КАЛИБРОВКА ЗОН — запускается ОДИН РАЗ, когда телефон уже
# закреплён в финальном положении над макетом.
#
# Что делает: показывает кадр с камеры телефона, вы кликаете
# мышкой по каждой из 6 лампочек по очереди, координаты
# сохраняются в файл zones.json для основного скрипта.
#
# УСТАНОВКА БИБЛИОТЕК (один раз в терминале):
#   pip install opencv-python
# ============================================================

import cv2
import json

# адрес из приложения DroidCam на вашем телефоне
VIDEO_URL = "http://10.12.3.163:4747/video"

NUM_ZONES = 20
# Порядок кликов ОБЯЗАТЕЛЕН:
# сначала 14 ламп на PCA9685 (каналы 0,1,3,4,5,6,7,8,9,11,12,13,14,15 по порядку,
# каналы 2 и 10 пропущены),
# потом 6 ламп на самой Arduino (пины D3,D5,D6,D9,D10,D11 по порядку)

points = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < NUM_ZONES:
        points.append((x, y))
        zone_idx = len(points) - 1   # 0-based номер зоны по порядку клика

        if zone_idx < 14:
            # вычисляем номер канала PCA9685, пропуская 2 и 10
            channel = zone_idx
            if channel >= 2:
                channel += 1
            if channel >= 10:
                channel += 1
            print(f"Зона {len(points)} отмечена (канал PCA9685 №{channel}): ({x}, {y})")
        else:
            arduino_pins = [3, 5, 6, 9, 10, 11]
            pin = arduino_pins[zone_idx - 14]
            print(f"Зона {len(points)} отмечена (пин Arduino D{pin}): ({x}, {y})")


cap = cv2.VideoCapture(VIDEO_URL)
ret, frame = cap.read()

if not ret:
    print("ОШИБКА: не удалось получить кадр с камеры.")
    print("Проверьте, что телефон и ноутбук в одной Wi-Fi сети,")
    print("и что VIDEO_URL указан правильно (адрес из приложения на телефоне).")
    exit()

cv2.namedWindow("Kalibrovka")
cv2.setMouseCallback("Kalibrovka", click_event)

print("Кликните мышкой по каждой из 20 лампочек СТРОГО по порядку:")
print("  1-14  = лампы на PCA9685 (каналы 0,1,3,4,5,6,7,8,9,11,12,13,14,15)")
print("  15-20 = лампы напрямую на Arduino (пины D3,D5,D6,D9,D10,D11)")
print("Нажмите 'q' в любой момент, чтобы выйти без сохранения.")

while True:
    display = frame.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(display, str(i + 1), (x + 12, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Kalibrovka", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or len(points) == NUM_ZONES:
        break

cap.release()
cv2.destroyAllWindows()

if len(points) == NUM_ZONES:
    with open("zones.json", "w") as f:
        json.dump(points, f)
    print("\nГотово! Координаты сохранены в zones.json:")
    print(points)
else:
    print("\nВы отметили не все 15 точек, файл не сохранён. Запустите скрипт заново.")
