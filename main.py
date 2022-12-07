import cv2
import numpy as np

def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(999)
    cv2.destroyAllWindows()

    green = np.uint8([[[0, 255, 0]]])
    green_hsv = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    print(green_hsv)

cap = cv2.VideoCapture("22.m4v")
template = cv2.imread("21.1.jpg", cv2.IMWRITE_TIFF_YDPI )
w, h = template.shape[::-1]


while True:
    _, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #########
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    viewImage(hsv_img)  # 1

    green_low = np.array([45, 100, 50])
    green_high = np.array([75, 255, 255])
    curr_mask = cv2.inRange(hsv_img, green_low, green_high)
    hsv_img[curr_mask > 0] = ([75, 255, 200])
    viewImage(hsv_img)  # 2

    # Преобразование HSV-изображения к оттенкам серого для дальнейшего оконтуривания
    RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    viewImage(gray)  # 3

    ret, threshold = cv2.threshold(gray, 90, 255, 0)
    viewImage(threshold)  # 4

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
    viewImage(image)  # 5
    #######
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.6)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 1:
        break

print(frame)
