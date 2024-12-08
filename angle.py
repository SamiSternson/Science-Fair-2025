import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import csv

cam_port = 0


def gcd(a, b):

    if a == 0:
        return b
    return gcd(b % a, a)


def measure_angle_right():
    cam = cv2.VideoCapture(cam_port)
    result, image = cam.read()
    print(result)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.medianBlur(gray_image[50:480, 200:640], 15)
    img_blur = cv2.GaussianBlur(gray_image, (15, 15), 0)
    dges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    zipped = np.dstack((gray_image, dges))
    x = []
    y = []
    height, width = gray_image.shape
    print(height, width)
    top = None
    for y_i, row in enumerate(dges):
        for x_i, pixel in enumerate(row):
            if pixel > 0.95:
                if top == None:
                    top = y_i
                    left = x_i
                    break
    dges = dges[top:-1, left:-1]
    for y_i, row in enumerate(dges):
        for x_i, pixel in enumerate(row):
            if pixel > 0.95:
                x.append(x_i)
                y.append(abs(y_i - len(zipped)))
    cv2.imshow("Images", dges)
    cv2.imshow("Color", image)
    cv2.imshow("gray", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    slope, intercept, r, p, std_err = stats.linregress(x, y)

    tick_size = gcd(height, width)
    line_x = np.linspace(0, width)
    plt.plot(line_x, line_x * slope + intercept)
    plt.scatter(x, y)
    angle_of_rep = np.rad2deg(np.arctan(abs(slope)))
    plt.yticks([i * tick_size for i in range((height // tick_size) + 1)])
    plt.show()
    return angle_of_rep


def measure_angle_left():
    cam = cv2.VideoCapture(cam_port)
    result, image = cam.read()
    print(result)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.medianBlur(gray_image[50:480, 0:400], 15)
    img_blur = cv2.GaussianBlur(gray_image, (15, 15), 0)
    dges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    zipped = np.dstack((gray_image, dges))
    x = []
    y = []
    height, width = gray_image.shape
    print(height, width)
    top = None
    for y_i, row in enumerate(dges):
        for x_i, pixel in enumerate(row):
            if pixel > 0.95:
                if top == None:
                    top = y_i
                    right = x_i
                    break
    dges = dges[top:-1, 0:right]
    for y_i, row in enumerate(dges):
        for x_i, pixel in enumerate(row):
            if pixel > 0.95:
                x.append(x_i)
                y.append(abs(y_i - len(zipped)))
    cv2.imshow("Images", dges)
    cv2.imshow("Color", image)
    cv2.imshow("gray", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    slope, intercept, r, p, std_err = stats.linregress(x, y)

    tick_size = gcd(height, width)
    line_x = np.linspace(0, width)
    plt.plot(line_x, line_x * slope + intercept)
    plt.scatter(x, y)
    angle_of_rep = np.rad2deg(np.arctan(abs(slope)))
    plt.yticks([i * tick_size for i in range((height // tick_size) + 1)])
    plt.show()
    return angle_of_rep


print("File Name?")
csv_file_path = input()
with open(csv_file_path, "r") as file:
    reader = csv.reader(file)
    data = list(reader)
    while True:
        command = input().lower()
        if command == "kill" or command == "done":
            break
        elif command == "new column" or command == "nc":
            print("Column Name?")
            cname = input()
            data[0] = cname
        elif command == "new entry" or command == "ne":
            repose = (measure_angle_right() + measure_angle_left()) / 2
            data[cname].append(repose)
with open(csv_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)
