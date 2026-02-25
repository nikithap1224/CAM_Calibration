import cv2
import numpy as np

# 1. Load your saved Brio 100 calibration data
with np.load('Brio100_calib.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# 2. Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# 3. Define the physical size of your marker (IMPORTANT for distance)
MARKER_SIZE = 5.0  # centimeters

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw the detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        # rvec = rotation vector, tvec = translation vector (distance)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)

        for i in range(len(ids)):
            # Draw axis for each marker to show 3D orientation
            cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 3)
            
            # Calculate distance (the Z-axis value of the translation vector)
            distance = tvecs[i][0][2]
            cv2.putText(frame, f"Dist: {distance:.2f}cm", (10, 30 + (i*30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Brio 100 - ArUco Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
