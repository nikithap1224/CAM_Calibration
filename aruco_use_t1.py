import numpy as np
import cv2

# Load the saved calibration data
with np.load('Brio100_calib.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

print("Loaded Camera Matrix:\n", mtx)

# Start your webcam application here...
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # Use mtx and dist to undistort the live feed
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)
    cv2.imshow('Calibrated Brio 100 Feed', undistorted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
