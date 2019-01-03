import numpy as np
import cv2
import sys
import time
import matplotlib.pyplot as plt
from math import sqrt
# 2.7g ball: kp: 1617.05, ki: 1848.74, kd: 231.09
class PIDController:
    def __init__(self, target_pos):
        self.target_pos = target_pos
        self.Kp = 4201.68 #2563.03
        self.Ki = 3571.43#2878.15
        self.Kd = 252.10#338.15
        self.bias = 0.0 #static
        self.last_error = 0.0
        self.Iterm = 0.0
        
        self.frame_counter = 0.0

        return
    def reset(self):
        self.Pterm = 0.0
        self.Iterm = 0.0
        self.Dterm = 0.0
        self.last_error = 0.0
        self.output = 0.0
        return
    
    def get_fan_rpm(self, image_frame=None):
        threshold = 1000.0
        self.frame_counter += 1
        temp = self.detect_ball(image_frame)
        vertical_ball_position = (483-(temp + 1))/483
        print(vertical_ball_position)
        error = self.target_pos - vertical_ball_position
        delta_error = error - self.last_error
        delta_time = 1/60 #60fps
            
        self.Pterm = self.Kp * error
            
        self.Iterm += error * delta_time
        if (self.Iterm < -threshold):
            self.Iterm = -threshold
        elif(self.Iterm > threshold):
            self.Iterm = threshold


        self.Dterm = delta_error / delta_time
        
        # Remember last time and last error for next calculation
        self.last_error = error

        output = self.Pterm + (self.Ki * self.Iterm) + (self.Kd * self.Dterm)
            
            #print('current error: ', error)
            #print('vertical position: ', vertical_ball_position)
            #print('pid: ', output)
            #print('P: ', self.Pterm, 'I: ', self.Iterm, 'D: ', self.Dterm)
        print(output)
        return output, vertical_ball_position



    def detect_ball(self, frame):
        bgr_color = 75, 65, 160
        color_threshold = 50
        
        hsv_color = cv2.cvtColor( np.uint8([[bgr_color]] ), cv2.COLOR_BGR2HSV)[0][0]
        HSV_lower = np.array([hsv_color[0] - color_threshold, hsv_color[1] - color_threshold, hsv_color[2] - color_threshold])
        HSV_upper = np.array([hsv_color[0] + color_threshold, hsv_color[1] + color_threshold, hsv_color[2] + color_threshold])
        x, y, radius = -1, -1, -1
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)
        mask = cv2.erode(mask, None, iterations= 1)#5
        mask = cv2.dilate(mask, None, iterations= 5)#multiple contours
        
        im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = (-1, -1)
        
        # only proceed if at least one contour was found
        if len(contours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            #print(radius)
            #print (x)
            M = cv2.moments(mask)
            horizontal = int(M["m10"] / M["m00"])
            vertical = int(M["m01"] / M["m00"])
            center = (horizontal, vertical)
            #print(center)
            # check that the radius is larger than some threshold
            if radius > 0:
                #outline ball
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                #show ball center
                cv2.circle(frame, center, 3, (0, 255, 0), -1)
        return center[1]
