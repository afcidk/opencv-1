from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import reshape, save
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import random

dot_array = []
dot_cnt = 0

def load_image(obj):
    img = cv2.imread('images/dog.bmp')
    cv2.imshow(image, img)
    h, w = img.shape[:2]
    print(f'Height: {h}\nWeight: {w}')

def color_conversion(obj):
    img = cv2.imread('images/color.png')
    new_img = img[:,:,[1,2,0]]
    cv2.imshow('original', img)
    cv2.imshow('conversion', new_img)
    print("color_conversion")

def image_flipping(obj):
    img = cv2.imread('images/dog.bmp')
    new_img = cv2.flip(img, 1)
    cv2.imshow('original', img)
    cv2.imshow('flipped', new_img)
    print("image_flipping")

def blending(obj):
    img1 = cv2.imread('images/dog.bmp')
    img2 = cv2.flip(img1, 1)
    cv2.imshow('blending', img2)
    cv2.createTrackbar('X', 'blending', 0, 100, blending_onchange)

    print("blending")

def blending_onchange(x):
    img1 = cv2.imread('images/dog.bmp')
    img2 = cv2.flip(img1, 1)
    img = cv2.addWeighted(img1, x/100, img2, (100-x)/100, 0.0)
    cv2.imshow('blending', img)

def global_threshold(obj):
    img = cv2.cvtColor(cv2.imread('images/QR.png'), cv2.COLOR_BGR2GRAY)
    _, new_img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    cv2.imshow('global threshold', new_img)
    print("global_threshold")

def local_threshold(obj):
    img = cv2.cvtColor(cv2.imread('images/QR.png'), cv2.COLOR_BGR2GRAY)
    new_img = cv2.adaptiveThreshold(img, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_MEAN_C, 19, -1)
    cv2.imshow('local threshold', new_img)
    print("local_threshold")

def pt(obj):
    global dot_array, dot_cnt
    dot_array, dot_cnt = [], 0
    img = cv2.imread("images/OriginalPerspective.png")
    cv2.imshow('op', img)
    cv2.setMouseCallback('op', draw_dots, img)
    print("pt")

def pt_real(img):
    global dot_array
    pts1 = np.float32(dot_array)
    pts2 = np.float32([[0,0], [500,0], [500,500], [0,500]])
    mat = cv2.getPerspectiveTransform(pts1, pts2)
    new_img = cv2.warpPerspective(img, mat, (500, 500))
    cv2.imshow('Perspective transform', new_img)

def draw_dots(event, x, y, flag, param):
    global dot_cnt, dot_array
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 10, (0,255,0), -1)
        cv2.imshow('op', img)
        dot_cnt += 1
        dot_array += [[x,y]]
        if dot_cnt == 4:
            pt_real(img)

def gaussian(obj):
    img = cv2.imread("images/School.jpg")
    cv2.imshow('School.jpg', img)
    # To grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_filter = compute_filter(sigma=0.707)
    g_img = apply_filter(gray_img, g_filter)

    cv2.imshow('gaussian filter', g_img)
    obj.g_img = g_img

def compute_filter(sigma):
    arr = []
    sigma2 = 2*sigma*sigma
    for i in range(-1,2):
        for j in range(-1,2):
            arr += [math.exp(-(i*i+j*j)/sigma2)/(math.pi*sigma2)]
    
    arr_sum = sum(arr)
    ret_arr = np.zeros((3,3))
    for i in range(len(arr)):
        ret_arr[i//3][i%3] = arr[i]/arr_sum

    return ret_arr

def sobel_x(obj):
    try:
        img = obj.g_img
    except:
        gaussian(obj)
        img = obj.g_img


    gx_arr = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sx_img = apply_filter(img, gx_arr)
    cv2.imshow('sobel_x', sx_img)
    obj.sx_img = sx_img

def apply_filter(img, kernel):

    y, x = img.shape
    m, n = kernel.shape
    y = y-m+1
    x = x-m+1
    ret_img = np.zeros((y,x), np.uint8)

    for i in range(y):
        for j in range(x):
            ret_img[i][j] = abs(np.sum(img[i:i+m, j:j+m]*kernel))
    return ret_img

def sobel_y(obj):
    try:
        img = obj.g_img
    except:
        gaussian(obj)
        img = obj.g_img

    gy_arr = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sy_img = apply_filter(img, gy_arr)
    cv2.imshow('sobel_y', sy_img)
    obj.sy_img = sy_img

def rst(obj):
    angle = float(obj.angle.text())
    scale = float(obj.scale.text())
    tx = int(obj.tx.text())
    ty = int(obj.ty.text())
    img = cv2.imread("images/OriginalTransform.png")
    rows, cols = img.shape[:-1]
    M = cv2.getRotationMatrix2D((130, 125), angle, scale)
    new_img = cv2.warpAffine(img, M, (cols, rows))
    H = np.float32([[1,0,tx],[0,1,ty]])
    new_img = cv2.warpAffine(new_img, H, (cols,rows))

    cv2.imshow("Original", img)
    cv2.imshow("After rotation, scale, translation", new_img)

def ok(obj):
    print("ok")

def cancel(obj):
    exit(0)

def magnitude(obj):
    try:
        img_x = obj.sx_img
    except:
        sobel_x(obj)
        img_x = obj.sx_img

    try:
        img_y = obj.sy_img
    except:
        sobel_y(obj)
        img_y = obj.sy_img

    img_mag = np.zeros((img_x.shape[0], img_x.shape[1], 1), np.uint8)
    for i in range(img_mag.shape[0]):
        for j in range(img_mag.shape[1]):
            img_mag[i][j] = math.sqrt(img_x[i][j]**2+img_y[i][j]**2)

    cv2.imshow('magnitude', img_mag)

def show_train_image(obj):
    x_train = obj.data_train.data
    y_train = obj.data_train.targets
    size, width, height = x_train.shape

    select = random.sample(range(0, size), 10)
    train_img = np.zeros((height+30, width*10, 1), np.uint8)
    train_img.fill(255)
    for i in range(height):
        for k in range(10):
            for j in range(width):
                train_img[i][k*width+j] = x_train[select[k]][i][j]

    for i in range(10):
        cv2.putText(train_img, str(y_train[select[i]].item()), (i*width+2, height+25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
    cv2.imshow('train img sample', train_img)

def show_hyper(obj):
    print(f"""
    hyperparameters:
    batch size: {obj.batch_size}
    learning rate: {obj.learning_rate}
    optimizer: {obj.opt}
    """)

def train_1(obj):
    obj.train(1)
    plt.plot(obj.loss_list)
    plt.show()

def show_train_result(obj):
    if obj.loaded == True:
        cv2.imshow('result', cv2.imread('result.png'))
        return 
    else:
        for i in range(1,51):
            obj.test_and_train(i)
        save(obj.net.state_dict(), 'model_params.pkl')
        obj.loaded = True

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(obj.acc_train_epoch)
    plt.plot(obj.acc_test_epoch)
    plt.title('Accuracy')
    plt.legend(['Training', 'Testing'])
    plt.ylabel('%')
    plt.subplot(2,1,2)
    plt.plot(obj.loss_epoch)
    plt.ylabel('Loss')
    plt.savefig('result.png')
    plt.show()

def inference(obj):
    if obj.loaded == False:
        for i in range(1,51):
            obj.test_and_train(i)
        save(obj.net.state_dict(), 'model_params.pkl')
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(obj.acc_train_epoch)
        plt.plot(obj.acc_test_epoch)
        plt.title('Accuracy')
        plt.legend(['Training', 'Testing'])
        plt.ylabel('%')
        plt.subplot(2,1,2)
        plt.plot(obj.loss_epoch)
        plt.ylabel('Loss')
        plt.savefig('result.png')
        obj.loaded = True

    idx = int(obj.test_index.text())
    value = obj.data_test.targets[idx].item()
    img = obj.data_test.data[idx]
    img = transforms.ToPILImage()(img)
    img = obj.compose(img)
    img = img[None]

    obj.net.eval()
    output = obj.net(img)
    pred = output.detach().max(1)[1]
    plt.figure()
    plt.xticks(np.arange(9)+1)
    zeros = np.zeros((9), np.uint8)
    zeros[value-1] = 1
    plt.bar(np.arange(9)+1, zeros)

    plt.figure()
    plt.imshow(transforms.ToPILImage()(img.squeeze(0)))
    plt.show()

