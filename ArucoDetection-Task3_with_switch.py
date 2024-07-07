import cv2
import numpy as np
import math
import time
from djitellopy import Tello

# Initialize camera settings for PC camera
camera_matrix_pc = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs_pc = np.zeros((5, 1), dtype=np.float32)

# Initialize camera settings for drone camera
camera_matrix_drone = np.array(
    [
        [921.170702, 0.000000, 459.904354],
        [0.000000, 919.018377, 351.238301],
        [0.000000, 0.000000, 1.000000],
    ]
)
dist_coeffs_drone = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

# Start with PC camera
camera_matrix = camera_matrix_pc
dist_coeffs = dist_coeffs_pc
use_drone = False

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_params = cv2.aruco.DetectorParameters()

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

def draw_axis(frame, tvec):
    try:
        axis_length = 0.1
        axis_points = np.float32(
            [
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, -axis_length],
            ]
        ).reshape(-1, 3)

        rvecs, _ = cv2.Rodrigues(np.zeros(3))
        axis_points, _ = cv2.projectPoints(
            axis_points, rvecs, tvec, camera_matrix, dist_coeffs
        )

        axis_points = axis_points.reshape(-1, 2).astype(int)

        frame = cv2.line(
            frame, tuple(axis_points[0]), tuple(axis_points[1]), (0, 0, 255), 2
        )
        frame = cv2.line(
            frame, tuple(axis_points[0]), tuple(axis_points[2]), (0, 255, 0), 2
        )
        frame = cv2.line(
            frame, tuple(axis_points[0]), tuple(axis_points[3]), (255, 0, 0), 2
        )

        text = f"ArUco marker Position x={tvec[0]:.2f} y={tvec[1]:.2f} z={tvec[2]:.2f}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    except Exception:
        pass

    return frame

initial_data = None
alignment_in_progress = False
last_command = None
last_command_time = time.time()
frame_data = []
recording = False
avg_positions = []
current_avg_index = 0

def get_movement_command(initial, current, position_index):
    tvec_diff = current["tvec"] - initial["tvec"]
    yaw_diff = current["yaw"] - initial["yaw"]
    pitch_diff = current["pitch"] - initial["pitch"]
    roll_diff = current["roll"] - initial["roll"]

    if np.abs(tvec_diff[2]) > 0.1:
        return "Move forward" if tvec_diff[2] > 0 else "Move backward"

    if use_drone:
        if np.abs(tvec_diff[0]) > 0.02:
            return "Move right" if tvec_diff[0] > 0 else "Move left"
    else:
        if np.abs(tvec_diff[0]) > 0.02:
            return "Move left" if tvec_diff[0] > 0 else "Move right"

    if np.abs(tvec_diff[1]) > 0.02:
        return "Move down" if tvec_diff[1] > 0 else "Move up"

    return f"Successfully returned to position {position_index + 1}!"

def calculate_average_positions(data, splits=5):
    if len(data) < splits:
        return []

    split_size = len(data) // splits
    avg_positions = []

    for i in range(splits):
        split_data = data[i * split_size:(i + 1) * split_size]

        # Count the occurrences of each QR code in the split
        qr_count = {}
        for frame in split_data:
            qr_id = frame["qr_id"]
            if qr_id in qr_count:
                qr_count[qr_id] += 1
            else:
                qr_count[qr_id] = 1

        # Find the QR code with the maximum occurrences
        selected_qr_id = max(qr_count, key=qr_count.get)
        selected_qr_frames = [frame for frame in split_data if frame["qr_id"] == selected_qr_id]

        # Calculate the average position based on the selected QR code frames
        avg_tvec = np.mean([d["tvec"] for d in selected_qr_frames], axis=0)
        avg_yaw = np.mean([d["yaw"] for d in selected_qr_frames])
        avg_pitch = np.mean([d["pitch"] for d in selected_qr_frames])
        avg_roll = np.mean([d["roll"] for d in selected_qr_frames])

        avg_positions.append({
            "tvec": avg_tvec,
            "yaw": avg_yaw,
            "pitch": avg_pitch,
            "roll": avg_roll,
        })

    avg_positions.reverse()

    return avg_positions


print(
    "Press 'c' to start/stop recording frames. Press 'm' to start the alignment process. Press 's' to switch camera feed. Press 'q' to quit."
)

cap = cv2.VideoCapture(0)
tello = None

while True:
    if use_drone:
        frame = tello.get_frame_read().frame
    else:
        ret, frame = cap.read()
        if not ret:
            break

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

            frame = draw_axis(frame, tvec)

            if recording:
                if len(frame_data) == 0 or (time.time() - frame_data[-1]["time"] >= 1):
                    frame_data.append({"time": time.time(), **current_data})

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

    # Mirror the camera feed for PC display
    if not use_drone:
        frame = cv2.flip(frame, 1)

    cv2.imshow("Tello Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):  # 'c' key is pressed to start/stop recording frames
        recording = not recording
        if not recording:
            print("Stopped recording frames.")
        else:
            print("Started recording frames.")
    elif key == ord("m"):  # 'm' key is pressed to start alignment
        if frame_data:
            avg_positions = calculate_average_positions(frame_data, 6)
            if avg_positions:
                alignment_in_progress = True
                current_avg_index = 0
                last_command = None
                last_command_time = time.time()
                print("Starting alignment process. Follow the instructions.")
            else:
                print("Not enough data to calculate average positions.")
        else:
            print("Record some frames first by pressing 'c'.")
    elif key == ord("s"):  # 's' key is pressed to switch camera feed
        use_drone = not use_drone
        if use_drone:
            if cap is not None:
                cap.release()
                cap = None
            tello = Tello()
            tello.connect()
            tello.streamon()
            camera_matrix = camera_matrix_drone
            dist_coeffs = dist_coeffs_drone
            print("Switched to drone camera feed.")
        else:
            if tello is not None:
                tello.streamoff()
                tello = None
            cap = cv2.VideoCapture(0)
            camera_matrix = camera_matrix_pc
            dist_coeffs = dist_coeffs_pc
            print("Switched to PC camera feed.")
    elif key == ord("q"):  # 'q' key is pressed to quit
        break

# Stop the video stream
if tello is not None:
    tello.streamoff()
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
