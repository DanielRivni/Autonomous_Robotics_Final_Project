# Autonomous Robotics Final Project

This project is divided into four main parts, each focusing on different aspects of autonomous robotics using a TELLO EDU drone and Aruco markers for navigation and positioning.

## Part 1: Detection Range and Probability Analysis
In this part, we conducted experiments to measure the distance and probability at which our computer (DELL) and the TELLO EDU drone can detect Aruco codes. The results are compiled in the presentation (slide 3).

### Objectives:
- Compare the detection capabilities of the computer versus the drone.
- Analyze the maximum distance at which Aruco codes can be detected.
- Evaluate the probability of successful detection at varying distances.

## Part 2: Recreating a Drone's Track
This part involves manually recording a video using the drone's cameras and the computer, and then enabling the drone to reproduce the route taken in the video. The drone uses targets (Aruco codes) to help it position itself in space and can start from any point on the route, returning to the initial point of the video and following the route accurately.

### Objectives:
- Record a route manually with the drone and computer.
- Enable the drone to reproduce the recorded route using spatial targets.
- Ensure the drone can start from any point on the route and navigate back to the starting point of the video.

### Implementation:
- Manual video recording with the drone and computer.
- Use of targets for spatial positioning.
- Algorithm for route reproduction.

## Part 3: Consistent Take-off and Landing Route
In this part, we aim to ensure that the drone's take-off and landing occur on the same route. The route taken during take-off should be the same route the drone follows when landing.

### Objectives:
- Define a consistent route for take-off and landing.
- Ensure the drone follows the same path for both take-off and landing.

### Implementation:
- Route definition and mapping.
- Synchronization of take-off and landing routes.

## Part 4: Landing on a Moving Aruco code
The final part focuses on enabling the drone to land in front of a Aruco code, even if the Aruco code is moved (e.g., by a wire). The drone should adjust its landing position to align with the moving Aruco code.

### Objectives:
- Enable the drone to detect and land in front of a Aruco code.
- Ensure the drone can adjust its landing position if the Aruco code moves.

### Implementation:
- Aruco code detection algorithms.
- Dynamic adjustment of landing position.

## Conclusion
This project demonstrates various aspects of autonomous robotics, including detection range analysis, route reproduction, consistent navigation, and dynamic landing adjustments. Each part contributes to the overall goal of enhancing the autonomy and accuracy of drone navigation using Aruco markers.

### Authors
- Daniel Rivni
- Ori Elimelech
- Nitay levy
