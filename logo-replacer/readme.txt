# Logo Detection and Replacement System Documentation
Generated on: 2024-11-02 11:10:54

## System Overview
This system implements a sophisticated logo detection and replacement pipeline using computer vision techniques.

## Technical Specifications

### 1. Input/Output Information
- Input Frame: 1920x1080 pixels
- Old Logo Template: 90x127 pixels
- New Logo: 90x127 pixels

### 2. Detection Results
- Confidence Level: 0.9781
- Best Scale Found: 1.60x
- Detection Location: (1602, 67)
- Detection Time: 3245.16ms

### 3. Replacement Details
- Region Size: 144x203
- Replacement Time: 0.00ms

## Performance Metrics

### Processing Times
- Total Execution Time: 3313.11ms
- Logo Detection: 3245.16ms
- Logo Replacement: 0.00ms

### Resource Usage
- Initial Memory: 42.2MB
- Peak Memory: 50.6MB
- CPU Usage: 8.4%

### File Metrics
- Input Frame Size: 565.0KB
- Output Frame Size: 879.6KB
- Size Overhead: 55.7%

## Technical Implementation Details

### Detection Algorithm
- Method: Template Matching with Multi-Scale Detection
- Scale Range: 0.8x to 2.0x
- Scale Steps: 25
- Matching Algorithm: cv2.TM_CCOEFF_NORMED
- Confidence Threshold: 0.3

### Replacement Algorithm
- Method: Alpha Blending
- Transparency Handling: Yes
- Boundary Checking: Implemented
- Size Adjustment: Dynamic

### Error Handling
- Confidence Validation
- Boundary Checks
- Size Mismatch Handling
- File I/O Validation

## Process Summary
- Start Time: 2024-11-02T11:10:51.563185
- Process ID: 56044
- Status: Successful

## File Locations
- Input Frame: C:\Users\prady\Documents\Capstone\logo-replacer\frame_000001.png
- Old Logo Template: C:\Users\prady\Documents\Capstone\logo-replacer\old-pes-logo.png
- New Logo: C:\Users\prady\Documents\Capstone\logo-replacer\new-pes-logo.png
- Output Frame: C:\Users\prady\Documents\Capstone\logo-replacer\output_frame.png
- Metrics JSON: C:\Users\prady\Documents\Capstone\logo-replacer\single_frame_metrics.json
- Process Log: C:\Users\prady\Documents\Capstone\logo-replacer\single_frame_process.log



This Python script implements a logo replacement system for a single frame of a video, utilizing OpenCV for image processing, TQDM for progress tracking, and PSUtil for system metrics monitoring. Below is a detailed breakdown of what each part of the code does:

### Imports
- **cv2**: OpenCV library for computer vision tasks.
- **numpy**: Library for numerical operations, used for array manipulations.
- **time**: Used to measure execution time of various operations.
- **psutil**: Allows for retrieving information on system utilization (memory, CPU).
- **datetime**: Used to timestamp logs and metrics.
- **json**: Used to save metrics in JSON format.
- **pathlib**: Provides a convenient way to handle filesystem paths.
- **logging**: Used for logging information about the process.
- **tqdm**: Provides a progress bar for loops, enhancing user experience during long operations.

### Class Definition: `SingleFrameLogoReplacer`
- This class is responsible for handling the entire logo replacement process for a single frame.

#### `__init__` Method
- **Parameters**: Takes the paths for the input frame, old logo, new logo, and output directory.
- **Metrics Initialization**: A dictionary to hold various metrics such as start time, memory usage, and CPU usage.
- **Logging Configuration**: Sets up a logging file to record process details.

#### `analyze_images` Method
- Analyzes the input frame and logos.
- **Image Loading**: Loads the input frame, old logo, and new logo.
- **Metrics Recording**: Records dimensions, size in kilobytes, and the number of channels for each image.
- **Progress Tracking**: Utilizes TQDM to display progress for each image analysis step.

#### `detect_logo` Method
- Uses multi-scale template matching to detect the old logo in the frame.
- **Template Matching**: Tries different scales of the logo template (ranging from 0.8x to 2.0x) and computes a matching score using OpenCV’s `matchTemplate`.
- **Best Match Selection**: Tracks the best match confidence score and location.
- **Metrics Recording**: Records detection confidence, location, time taken for detection, and system metrics (memory and CPU usage).

#### `replace_logo` Method
- Replaces the detected old logo with the new logo.
- **Logo Resizing**: Resizes the new logo to match the dimensions of the detected old logo.
- **Alpha Blending**: Handles transparency by creating an alpha mask if the new logo has an alpha channel.
- **Region of Interest**: Ensures that the new logo is blended correctly into the frame at the detected location.
- **Metrics Recording**: Captures metrics for replacement time and resource usage.

#### `process_frame` Method
- The main workflow for processing a single frame.
- Calls the image analysis, logo detection, and logo replacement methods in sequence.
- **Error Handling**: Catches exceptions during processing and logs errors if the logo is not detected with sufficient confidence.
- **Output Saving**: Saves the processed frame and logs the overall processing metrics, including memory usage and total time taken.

#### `create_documentation` Method
- Generates a detailed documentation file summarizing the process.
- Contains sections on input/output details, detection results, performance metrics, and error handling strategies.
- Writes this documentation to a text file.

### `main` Function
- Instantiates the `SingleFrameLogoReplacer` class with specific file paths.
- Calls the `process_frame` method to execute the logo replacement process.

### Execution
- The script is designed to be run directly, where `main()` is called if the script is executed as a standalone program.

### Summary of Key Features
- **Modular Design**: The class-based approach organizes related functionality together.
- **Comprehensive Metrics**: Collects detailed metrics on performance, including memory usage and execution time, aiding in performance analysis.
- **Error Handling**: Incorporates checks to ensure robust processing, with logging for debugging purposes.
- **Documentation Generation**: Automatically creates documentation for each run, making it easy to track changes and performance.

This script is quite sophisticated and would serve well for tasks involving logo detection and replacement in images, with detailed monitoring and logging throughout the process. If you have any specific areas you want to modify or questions about, feel free to ask!