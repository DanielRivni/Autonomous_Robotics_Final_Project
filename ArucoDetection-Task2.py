import cv2
import numpy as np
import datetime
import math
import time
import os

# Dictionary of available ArUco marker types
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}

# Select the dictionary to use
selected_dict = ARUCO_DICTS["DICT_4X4_100"]

# Initialize the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(selected_dict)
aruco_params = cv2.aruco.DetectorParameters()

# Camera calibration parameters (assumed to be known)
camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Helper function to convert rotation matrix to Euler angles
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

# Initialize webcam capture to get dimensions
cap_webcam = cv2.VideoCapture(0)
ret, frame_webcam = cap_webcam.read()
cap_webcam.release()
if not ret:
    raise Exception("Could not read from webcam.")
webcam_height, webcam_width = frame_webcam.shape[:2]

# Initialize video capture from file
video_path = 'C://temp//vid_3.mp4'
if os.path.exists(video_path):
    print("The video file exists.")
else:
    print("The video file does not exist.")
cap = cv2.VideoCapture(video_path)

# List to store frame data
frame_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match webcam dimensions
    frame = cv2.resize(frame, (webcam_width, webcam_height))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.05, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs
        )

        for i, corner in enumerate(corners):
            qr_id = ids[i][0]
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]

            dist = np.linalg.norm(tvec)
            rmat, _ = cv2.Rodrigues(rvec)
            yaw, pitch, roll = rotationMatrixToEulerAngles(rmat)

            current_data = {
                "qr_id": qr_id,
                "dist": dist,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "tvec": tvec.tolist(),
            }

            frame_data.append({"time": datetime.datetime.now().isoformat(), **current_data})

cap.release()

# Function to calculate average positions
def calculate_average_positions(data, splits=5):
    if len(data) < splits:
        return []

    split_size = len(data) // splits
    avg_positions = []

    for i in range(splits):
        split_data = data[i * split_size:(i + 1) * split_size]
        avg_tvec = np.mean([d["tvec"] for d in split_data], axis=0)
        avg_yaw = np.mean([d["yaw"] for d in split_data])
        avg_pitch = np.mean([d["pitch"] for d in split_data])
        avg_roll = np.mean([d["roll"] for d in split_data])

        avg_positions.append({
            "tvec": avg_tvec,
            "yaw": avg_yaw,
            "pitch": avg_pitch,
            "roll": avg_roll,
        })

    return avg_positions

# Function to get movement command
def get_movement_command(initial, current, position_index):
    tvec_diff = current["tvec"] - initial["tvec"]
    yaw_diff = current["yaw"] - initial["yaw"]
    pitch_diff = current["pitch"] - initial["pitch"]
    roll_diff = current["roll"] - initial["roll"]

    if np.abs(tvec_diff[2]) > 0.1:
        return "Move forward" if tvec_diff[2] > 0 else "Move backward"

    if np.abs(tvec_diff[0]) > 0.01:
        return "Move left" if tvec_diff[0] > 0 else "Move right"

    if np.abs(tvec_diff[1]) > 0.01:
        return "Move down" if tvec_diff[1] > 0 else "Move up"

    return f"Successfully returned to position {position_index + 1}!"

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# State variables
recording = False
alignment_in_progress = False
avg_positions = []
current_avg_index = 0
last_command = None
last_command_time = None

# Video writer for recording
out = None
recording_filename = None

print("Press 'r' to start/stop recording. Press 'm' to start the alignment process. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not recording:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.05, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs
            )

            for i, corner in enumerate(corners):
                qr_id = ids[i][0]
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                dist = np.linalg.norm(tvec)
                rmat, _ = cv2.Rodrigues(rvec)
                yaw, pitch, roll = rotationMatrixToEulerAngles(rmat)

                current_data = {
                    "qr_id": qr_id,
                    "dist": dist,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    "tvec": tvec,
                }

                if alignment_in_progress and avg_positions:
                    command = get_movement_command(avg_positions[current_avg_index], current_data, current_avg_index)
                    if command.startswith("Successfully"):
                        print(f"Position {current_avg_index + 1} reached!")
                        current_avg_index += 1
                        if current_avg_index >= len(avg_positions):
                            print("Successfully reached all positions!")
                            alignment_in_progress = False
                            break
                    else:
                        current_time = time.time()

                        if command != last_command and (
                            last_command_time is None
                            or (current_time - last_command_time) > 1
                        ):
                            print(command)
                            last_command = command
                            last_command_time = current_time
                        elif (
                            command == last_command
                            and (current_time - last_command_time) > 1
                        ):
                            print(command)
                            last_command_time = current_time

    # Record the frame if recording is in progress
    if recording and out is not None:
        out.write(frame)

    # Mirror the camera feed for display
    frame = cv2.flip(frame, 1)

    cv2.imshow("Webcam Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):  # 'r' key is pressed to start/stop recording
        if recording:
            print(f"Stopped recording. Video saved as {recording_filename}")
            recording = False
            out.release()
            out = None
        else:
            recording_filename = f'recorded_video_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            out = cv2.VideoWriter(
                recording_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (webcam_width, webcam_height)
            )
            print(f"Started recording. Video will be saved as {recording_filename}")
            recording = True
    elif key == ord("m"):  # 'm' key is pressed to start alignment
        if frame_data:
            avg_positions = calculate_average_positions(frame_data)
            if avg_positions:
                alignment_in_progress = True
                current_avg_index = 0
                last_command = None
                last_command_time = time.time()
                print("Starting alignment process. Follow the instructions.")
            else:
                print("Not enough data to calculate average positions.")
        else:
            print("Record some frames first by processing the video file.")
    elif key == ord("q"):  # 'q' key is pressed to quit
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
