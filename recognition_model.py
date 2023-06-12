import numpy as np
import pandas as pd

import torch
import cv2

from lpr_net import build_lprnet, rec_plate, CHARS

import os


def recognize_plate(img):
    try:
        best_weights = './yolo/best_964_521.pt'
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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        LPRnet = build_lprnet(lpr_max_len=9, phase=False, class_num=len(CHARS), dropout_rate=0)
        LPRnet.to(device)
        LPRnet.load_state_dict(torch.load('Final_LPRNet_model.pth', map_location=device))

        text = rec_plate(LPRnet, number_plate, device)

        return text
    except:
        return 'NaN'