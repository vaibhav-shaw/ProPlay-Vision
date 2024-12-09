# ProPlay Vision: A Comprehensive Player Tracking and Analysis System

## Overview
**ProPlay Vision** is an advanced project focused on detecting and tracking players, referees, and footballs in video footage using cutting-edge AI techniques. By leveraging the capabilities of **YOLO**, one of the leading object detection models, this project aims to provide detailed analysis and insights into football matches.

This system not only identifies and tracks objects but also clusters players into teams based on their t-shirt colors using **K-Means clustering**. It calculates crucial metrics like team ball possession percentage, individual player movement, and speed by integrating methods such as **optical flow** and **perspective transformation**. The system further quantifies these movements in real-world terms (meters) rather than pixels, providing actionable insights.

**ProPlay Vision** is an excellent project for both novice and experienced machine learning practitioners, offering hands-on experience with real-world problems and advanced computational techniques.

---

## Key Features
- **Player, Referee, and Ball Detection**  
  Utilizes the YOLO object detection model to accurately identify and track key elements in the video.

- **Team Identification via T-shirt Colors**  
  Implements **K-Means clustering** for pixel segmentation to classify players into teams based on the dominant colors of their t-shirts.

- **Ball Possession Analysis**  
  Computes the percentage of time each team possesses the ball, providing valuable match insights.

- **Player Movement Analysis**  
  Measures player movements using **optical flow** to account for camera dynamics between frames.

- **Perspective Transformation**  
  Applies perspective transformation to convert 2D pixel coordinates into real-world distances, enabling accurate calculations of player movement and field coverage in meters.

- **Speed and Distance Metrics**  
  Calculates the speed and total distance covered by each player during the match.

---

## Project Modules
The following modules and techniques are integral to the system:  
- **YOLO (You Only Look Once):** State-of-the-art object detection for real-time video analysis.  
- **K-Means Clustering:** Groups players into teams based on pixel color clustering for t-shirt detection.  
- **Optical Flow:** Captures camera motion between frames to ensure accurate tracking.  
- **Perspective Transformation:** Maps the 2D scene to a real-world perspective for precise measurement.  
- **Speed and Distance Analysis:** Tracks player dynamics to generate actionable statistics.

---

## Installation and Requirements
To set up and run the project:

1. Clone the repository:
   ```
   git clone https://github.com/vaibhav-shaw/ProPlay-Vision.git
   cd ProPlay-Vision
   ```
2. Install the required libraries using the provided requirements.txt file:
   ```
   pip install -r requirements.txt
   ```
   
## Project Applications
- `Player performance analysis in sports.`
- `Automated ball possession statistics for team strategy.`
- `Advanced video analytics for coaches and analysts.`
- `Real-world movement metrics for enhanced player evaluation.`

---

## Screenshot
![Screenshot](screenshot.png)

---

`ProPlay Vision` combines the power of computer vision and machine learning to deliver a comprehensive solution for football analytics. Dive in and explore its capabilities!
