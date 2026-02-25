import cv2
import numpy as np
import glob
import yaml
import os

# --- User settings ---
CHESSBOARD = (7, 7)          # inner corners per a chessboard row and column
SQUARE_SIZE_MM = 18.0        # size of a square in your defined unit (mm)
IMAGES_GLOB = 'calib_images/*.jpg'  # or capture from camera separately
OUTPUT_FILE = 'brio_calib.yaml'
# ----------------------

# prepare object points like (0,0,0), (1,0,0), ... scaled by square size
objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

images = glob.glob(IMAGES_GLOB)
if not images:
    raise SystemExit("No images found. Put calibration images in calib_images/ or change IMAGES_GLOB.")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners_refined)
        objpoints.append(objp)
        # draw and show for verification (optional)
        cv2.drawChessboardCorners(img, CHESSBOARD, corners_refined, ret)
        cv2.imshow('corners', img)
        cv2.waitKey(100)
cv2.destroyAllWindows()

if not objpoints:
    raise SystemExit("No chessboard corners detected. Try different images or pattern size.")

# Calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
mean_error = total_error / len(objpoints)

print("Calibration RMS:", ret)
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())
print("Mean reprojection error:", mean_error)

# Save to YAML
data = {
    'camera_matrix': camera_matrix.tolist(),
    'dist_coeff': dist_coeffs.tolist(),
    'reprojection_error': float(mean_error),
    'image_size': gray.shape[::-1]
}
with open(OUTPUT_FILE, 'w') as f:
    yaml.dump(data, f)

print("Saved calibration to", OUTPUT_FILE)
