# -*- coding: utf-8 -*-
import os

from PIL import Image
from PIL import ImageEnhance
import subprocess
import numpy as np
import cv2


def exe_cmd(cmd):
    s = subprocess.Popen(str(cmd), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    s.wait()
    print(s)


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def get_files(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(os.path.join(root, file))      # os.path.join 获取完整路径
    return L


def get_img_files(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                L.append(os.path.join(root, file))      # os.path.join 获取完整路径
    return L

# 写数据 flag:'w+'
def write_data(file_name, data, flag):
    with open(file_name, flag) as f:
        f.write(data)


# 读数据 flag:'r'
def read_data(file_name, flag):
    with open(file_name, flag) as f:
        return f.read()


# =====================================================================================
# 图片锐化
def image_sharpness(img, factor):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # image.show()
    enhancer_object = ImageEnhance.Sharpness(image)
    out = enhancer_object.enhance(factor)
    # out.show()
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


# =====================================================================================
# 图片旋转
def image_rotation(img, angle):
    rows, cols, _ = img.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


# 图片做仿射变换：
def image_affine(img, M, (w, h)):
    # rows, cols, ch = img.shape

    # pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    # pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    # M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (w, h))

    # plt.subplot(121), plt.imshow(img), plt.title('Input')
    # plt.subplot(122), plt.imshow(dst), plt.title('Output')
    return dst


# 图片做透视变换
def image_perspective(img):
    # img = cv2.imread('sudokusmall.png')
    # rows, cols, ch = img.shape

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (300, 300))
    #
    # plt.subplot(121), plt.imshow(img), plt.title('Input')
    # plt.subplot(122), plt.imshow(dst), plt.title('Output')
    # plt.show()
    return dst


# =====================================================================================
# 图片做高斯模糊
def image_gaussian_blur(img, kernel_size=(3, 3), sigma=1.5):
    img_gaussian = cv2.GaussianBlur(img, kernel_size, sigma)
    return img_gaussian


# 图片转灰度图
def iamge_to_gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


# =====================================================================================
# 图像用sobel算子进行边缘提取
def image_sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)                       # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # cv2.imshow("absX", absX)
    # cv2.imshow("absY", absY)
    return dst


# # 图像用laplacian算子进行边缘提取
# def image_laplacian(img):
#     cv2.Scharr()
#     x = cv2.Laplacian(img, cv2.CV_16S, 1, 0)
#     y = cv2.Laplacian(img, cv2.CV_16S, 0, 1)
#     absX = cv2.convertScaleAbs(x)                       # 转回uint8
#     absY = cv2.convertScaleAbs(y)
#     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#     cv2.imshow("absX", absX)
#     cv2.imshow("absY", absY)
#     return dst
#
#
# # 图像用Scharr算子进行边缘提取
# def image_scharr(img):
#     x = cv2.Scharr(img, cv2.CV_16S, 1, 0)
#     y = cv2.Scharr(img, cv2.CV_16S, 0, 1)
#     absX = cv2.convertScaleAbs(x)                       # 转回uint8
#     absY = cv2.convertScaleAbs(y)
#     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#     cv2.imshow("absX", absX)
#     cv2.imshow("absY", absY)
#     return dst


# 图像二值化
def image_threshold(img):
    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)    # 自适应阈值二值化
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    # ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    # ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    return thresh


# =====================================================================================
# 图片做开操作
def image_open(img, ksize=(17, 17)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    # 形态学操作
    # 第二个参数：要执行的形态学操作类型，这里是开操作
    binary = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("open", binary)
    return binary


# 图片做闭操作
def image_close(img, ksize=(17, 17)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    # 形态学操作
    # 第二个参数：要执行的形态学操作类型，这里是开操作
    binary = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("close", binary)
    return binary


# =====================================================================================
# 图片取轮廓
def image_findContours(img):
    # 第一个参数是寻找轮廓的图像；
    #
    # 第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
    # cv2.RETR_EXTERNAL 表示只检测外轮廓
    # cv2.RETR_LIST     检测的轮廓不建立等级关系
    # cv2.RETR_CCOMP    建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    # cv2.RETR_TREE     建立一个等级树结构的轮廓。
    #
    # 第三个参数method为轮廓的近似办法
    # cv2.CHAIN_APPROX_NONE   存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1 - x2），abs（y2 - y1）） == 1
    # cv2.CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    # cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh - Chinl chain 近似算法

    # image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return image, contours, hierarchy


# 查找图片的倾斜角度
def image_angle(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    print(angle)
    return angle


# =====================================================================================
def image_test(img):
    angle_list = []

    h, w, c = img.shape

    img_3 = image_gaussian_blur(img)
    # cv2.imshow('img_3', img_3)

    img_gray = iamge_to_gray(img_3)
    # cv2.imshow('img_gray', img_gray)

    img_sobel = image_sobel(img_gray)
    # cv2.imshow('img_sobel', img_sobel)

    img_thresh = image_threshold(img_sobel)
    # cv2.imshow('img_thresh', img_thresh)

    img_close = image_close(img_thresh, (7, 7))
    image, contours, hierarchy = image_findContours(img_close)

    # img_contours = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    # cv2.imshow("image_findContours", img_contours)

    for contour in contours:
        # 获取最小包围矩形
        rect = cv2.minAreaRect(contour)

        # 中心坐标
        x, y = rect[0]
        # 长宽,总有 width>=height
        width, height = rect[1]
        # 角度:[-90,0)
        angle = rect[2]

        if angle < -45:
            angle = -(90 + angle)
            height, width = rect[1]

        if width == 0 or height == 0:
            continue

        if width < 50 or width > 500:
            continue

        if height < 10 or height > 300:
            continue

        if float(width) / float(height) < 2:
            continue

        if float(width) / float(height) > 6:
            continue

        if float(width) / float(w) < 0.8:
            continue

        if angle < -40:
            continue

        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 5)
        cv2.rectangle(img, (int(x - width / 2), int(y - height / 2)), (int(x + width / 2), int(y + height / 2)),
                      (0, 0, 255), 2)
        cv2.drawContours(img, contour, -1, (255, 255, 0), 2)
        # print 'width=', width, 'height=', height, 'x=', x, 'y=', y, 'angle=', angle
        angle_list.append(angle)

    # cv2.imshow("contour", img)

    # cv2.imshow("rotation", img)
    # cv2.waitKey(0)
    return angle_list


def image_change():
    img = cv2.imread('/home/caiyueliang/deeplearning/lpr-service/capture_service/7.jpg')
    cv2.imshow('old', img)

    img_3 = image_gaussian_blur(img)
    cv2.imshow('img_3', img_3)

    img_gray = iamge_to_gray(img_3)
    cv2.imshow('img_gray', img_gray)

    img_sobel = image_sobel(img_gray)
    cv2.imshow('img_sobel', img_sobel)

    # img_gray = cv2.bitwise_not(img_gray)

    img_thresh = image_threshold(img_sobel)
    cv2.imshow('img_thresh', img_thresh)

    angle = image_angle(img_thresh)                             # 获取角度
    (h, w) = img_thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)             # 仿射变换，先获取变换矩阵

    img_rotated = image_affine(img, M, (w, h))                  # 仿射变换
    cv2.imshow('img_rotated', img_rotated)

    cv2.waitKey(0)
    return


# 获取RGB颜色的HSV值
def get_color_hsv():
    green = np.uint8([[[16, 79, 140]]])
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    print hsv_green
    return hsv_green


def image_detect(img):
    # Convert BGR to HSV

    img = image_gaussian_blur(img, kernel_size=(5, 5))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    # lower_blue = np.array([100, 50, 50])
    # upper_blue = np.array([130, 255, 255])
    # lower_blue = np.array([100, 60, 60])
    # upper_blue = np.array([130, 240, 240])
    lower_blue = np.array([100, 60, 60])
    upper_blue = np.array([130, 240, 220])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    # cv2.imshow('res', res)

    img_close = image_close(mask, (7, 7))
    cv2.imshow('img_close', img_close)
    image, contours, hierarchy = image_findContours(img_close)

    for contour in contours:
        # 获取最小包围矩形
        rect = cv2.minAreaRect(contour)

        # 中心坐标
        x, y = rect[0]
        # 长宽,总有 width>=height
        width, height = rect[1]
        # 角度:[-90,0)
        angle = rect[2]

        if angle < -45:
            angle = -(90 + angle)
            height, width = rect[1]

        if width == 0 or height == 0:
            continue

        if width < 50 or width > 500:
            continue

        if height < 10 or height > 300:
            continue

        if float(width) / float(height) < 2:
            continue

        if float(width) / float(height) > 6:
            continue

        if angle < -40:
            continue

        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 5)
        cv2.rectangle(img, (int(x - width / 2), int(y - height / 2)), (int(x + width / 2), int(y + height / 2)),
                      (0, 0, 255), 2)
        cv2.drawContours(img, contour, -1, (255, 255, 0), 2)
        print 'width=', width, 'height=', height, 'x=', x, 'y=', y, 'angle=', angle
        # angle_list.append(angle)

    cv2.imshow("rotation", img)
    return mask


# =====================================================================================
def fitLine_ransac(pts, zero_add=0):
    if len(pts) >= 2:
        [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((136 - x) * vy / vx) + y)
        # return lefty + 30 + zero_add, righty + 30 + zero_add
        return lefty + zero_add, righty + zero_add
    return 0, 0


# 精定位算法
def findContoursAndDrawBoundingBox(image_rgb):
    line_upper = []
    line_lower = []

    line_experiment = []
    grouped_rects = []
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

    # for k in np.linspace(-1.5, -0.2,10):
    for k in np.linspace(-50, -5, 15):

        # thresh_niblack = threshold_niblack(gray_image, window_size=21, k=k)
        # binary_niblack = gray_image > thresh_niblack
        # binary_niblack = binary_niblack.astype(np.uint8) * 255

        binary_niblack = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, k)
        # cv2.imshow("image1", binary_niblack)
        # cv2.waitKey(0)

        imagex, contours, hierarchy = cv2.findContours(binary_niblack.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bdbox = cv2.boundingRect(contour)
            # print('bdbox: ' + str(bdbox))
            if ((bdbox[3]/float(bdbox[2]) > 0.7 and bdbox[3]*bdbox[2] > 100 and bdbox[3]*bdbox[2] < 1200)
                    or (bdbox[3]/float(bdbox[2]) > 3 and bdbox[3]*bdbox[2] < 100)):
                # cv2.rectangle(rgb,(bdbox[0],bdbox[1]),(bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]),(255,0,0),1)
                line_upper.append([bdbox[0], bdbox[1]])
                line_lower.append([bdbox[0]+bdbox[2], bdbox[1]+bdbox[3]])

                line_experiment.append([bdbox[0], bdbox[1]])
                line_experiment.append([bdbox[0]+bdbox[2], bdbox[1]+bdbox[3]])
                # grouped_rects.append(bdbox)

    # rgb = cv2.copyMakeBorder(image_rgb, 30, 30, 0, 0, cv2.BORDER_REPLICATE)
    rgb = image_rgb
    # cv2.imshow('rgb', rgb)

    leftyA, rightyA = fitLine_ransac(np.array(line_lower), 20)
    # print(leftyA, rightyA)
    leftyB, rightyB = fitLine_ransac(np.array(line_upper), -20)
    # print(leftyB, rightyB)

    rows, cols = rgb.shape[:2]
    pts_map1 = np.float32([[cols - 1, rightyA], [0, leftyA], [cols - 1, rightyB], [0, leftyB]])
    pts_map2 = np.float32([[136, 36], [0, 36], [136, 0], [0, 0]])
    mat = cv2.getPerspectiveTransform(pts_map1, pts_map2)
    image = cv2.warpPerspective(rgb, mat, (136, 36), flags=cv2.INTER_CUBIC)
    # cv2.imshow('image', image)

    # cv2.waitKey(0)
    return image


# def findContours(image_rgb):
#
#     return img

def image_car_plate():
    files = get_files('/home/caiyueliang/deeplearning/lpr-service/capture_service/capture_image/20180718_094810/failed')
    save_path = '/home/caiyueliang/deeplearning/lpr-service/capture_service/capture_image/aaa'
    mkdir_if_not_exist(save_path)

    for file in files:
        img = cv2.imread(file)

        img_plate = image_detect(img)
        cv2.waitKey(0)
        # img_plate = findContoursAndDrawBoundingBox(img)
        # img_plate = image_test(img)

        cv2.imwrite(os.path.join(save_path, file.split('/')[-1]), img_plate)
    return


if __name__ == '__main__':
    # get_color_hsv()
    # image_car_plate()
    img = cv2.imread('/home/caiyueliang/deeplearning/lpr-service/capture_service/capture_image/test_1/0498627_闽DMF980.jpg')
    print(img)
    img1 = cv2.imread('/home/caiyueliang/deeplearning/lpr-service/capture_service/capture_image/test_1/498627_闽DMF980.jpg')
    print(img1)
