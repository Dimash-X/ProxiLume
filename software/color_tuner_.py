# ============================================================
# ЖИВАЯ НАСТРОЙКА ЦВЕТА — самый надёжный способ подобрать
# точные границы "красного" именно под вашу камеру и освещение.
#
# Открывает окно с 6 ползунками. Двигайте их, пока в окне
# "Maska" не станет БЕЛЫМ ровно ваш красный предмет,
# а всё остальное (фон, дерево стола, лампочки) — ЧЁРНЫМ.
#
# Когда добьётесь чистой белой фигуры под кнопкой/шапкой —
# нажмите 'p', и в консоли распечатаются готовые числа,
# которые нужно скопировать в main_tracker.py и calibrate_zones.py
#
# УСТАНОВКА (если ещё не делали):
#   python -m pip install opencv-python numpy
# ============================================================

import cv2
import numpy as np

VIDEO_URL = "http://10.12.3.163:4747/video"   # ваш адрес DroidCam


def nothing(x):
    pass


cap = cv2.VideoCapture(VIDEO_URL)

if not cap.isOpened():
    print("ОШИБКА: не удалось подключиться к камере.")
    print("Проверьте: телефон и ноутбук в одной Wi-Fi сети,")
    print("приложение DroidCam запущено и показывает IP/Port,")
    print("VIDEO_URL в этом файле совпадает с тем, что на экране телефона.")
    exit()

cv2.namedWindow("Nastroyka")
cv2.createTrackbar("H min", "Nastroyka", 0, 180, nothing)
cv2.createTrackbar("H max", "Nastroyka", 10, 180, nothing)
cv2.createTrackbar("S min", "Nastroyka", 120, 255, nothing)
cv2.createTrackbar("S max", "Nastroyka", 255, 255, nothing)
cv2.createTrackbar("V min", "Nastroyka", 70, 255, nothing)
cv2.createTrackbar("V max", "Nastroyka", 255, 255, nothing)

print("Двигайте ползунки, пока предмет не станет чисто белым на маске.")
print("Нажмите 'p' чтобы распечатать текущие значения.")
print("Нажмите 'q' чтобы выйти.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Кадр не получен, пробуем снова...")
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H min", "Nastroyka")
    h_max = cv2.getTrackbarPos("H max", "Nastroyka")
    s_min = cv2.getTrackbarPos("S min", "Nastroyka")
    s_max = cv2.getTrackbarPos("S max", "Nastroyka")
    v_min = cv2.getTrackbarPos("V min", "Nastroyka")
    v_max = cv2.getTrackbarPos("V max", "Nastroyka")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Video", frame)
    cv2.imshow("Maska", mask)
    cv2.imshow("Rezultat (chto naideno)", result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        print("\n--- ТЕКУЩИЕ ЗНАЧЕНИЯ ---")
        print(f"lower_red1 = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper_red1 = np.array([{h_max}, {s_max}, {v_max}])")
        print("(если красный у вас 'чистый' и попадает в один диапазон,")
        print(" можно использовать только этот один диапазон в main_tracker.py)")
        print("-------------------------\n")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
