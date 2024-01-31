from typing import Any
import cv2
import numpy as np

from grip import GripPipeline

def processFrame(frame: cv2.Mat):
	grip = GripPipeline()
	
	grip.process(frame)
	contours = grip.find_contours_output
	if contours is not None and len(contours) > 0:
		cv2.drawContours(frame, contours, -1, (255, 0, 0), 10)

		# get the biggest contour
		cnt = max(contours, key=cv2.contourArea)
		# determine the most extreme points along the contour
		extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
		extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
		extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
		extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
		print(extLeft, extRight, extTop, extBot)
		cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
		cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
		cv2.circle(frame, extTop, 8, (255, 0, 255), -1)
		cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

	black = np.zeros(frame.shape, np.uint8)
	cv2.imshow("fart", frame)
	cv2.waitKey(1)
	

# def main():
# 	cam = cv2.VideoCapture(0)
# 	while True:
# 		success, frame = cam.read()
# 		processFrame(frame)

# main()