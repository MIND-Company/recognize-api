import numpy as np
import pandas as pd

import torch
import cv2

import os

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\Python\tesseract\tesseract.exe' # Изменить путь на свой

def recognize_plate(img):
    try:
        best_weights = './yolo/main_best_964_51.pt'
        model = torch.hub.load('ultralytics/yolov5', 'custom', best_weights)

        im = cv2.imread(img)
        carplate_img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        results = model(carplate_img_rgb)
        # Getting coordinates of license plate
        results_df = results.pandas().xyxy[0].loc[0]
        x_min = int(results_df['xmin'])
        x_max = int(results_df['xmax'])
        y_min = int(results_df['ymin'])
        y_max = int(results_df['ymax'])
        # Cropping license plate from image ""
        number_plate = carplate_img_rgb[y_min:y_max, x_min:x_max]
        number_plate = cv2.resize(number_plate, (int(3 * number_plate.shape[1]), int(3 * number_plate.shape[0])))

        text = pytesseract.image_to_string(number_plate,
                                           config=f'--psm 13 --oem 1 '
                                                  f'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        return text
    except:
        return 'NaN'