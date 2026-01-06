================================================================================
                    CS 5330 - Pattern Recognition and Computer Vision
                         Project 1: Video Special Effects
================================================================================

Name: Joseph Defendre
Date: January 5 2026
Course: CS 5330 Pattern Recognition and Computer Vision
Project: Project 1 - Video Special Effects

================================================================================
                                  DESCRIPTION
================================================================================

This project implements a real-time video processing application that applies
various image filters and special effects to live video from a webcam or to
static images. The application demonstrates fundamental computer vision
techniques including color space manipulation, convolution-based filtering,
edge detection, and face detection.

Key features include:
- Real-time video capture and processing
- Multiple image filters (greyscale, sepia, blur, emboss, etc.)
- Edge detection using Sobel operators
- Cartoon effect using blur and quantization
- Face detection using Haar cascades
- Depth estimation using ONNX neural network model (optional)
- Vignette and negative effects
- Frame saving capability

================================================================================
                            BUILDING INSTRUCTIONS
================================================================================

Prerequisites:
--------------
- OpenCV 4.x (required)
- ONNX Runtime (optional, for depth estimation feature)
- C++ compiler with C++11 support
- Make build system

Build Steps:
------------
1. Navigate to the source directory:
   cd src

2. Build all targets:
   make all

3. The compiled binaries will be placed in the bin/ directory.

To clean the build:
   make clean

================================================================================
                             RUNNING INSTRUCTIONS
================================================================================

Image Display:
--------------
   ./bin/imgDisplay [image_path]

   Displays a static image with the ability to apply filters.
   If no image path is provided, the program will use a default image.

Video Processing:
-----------------
   ./bin/vid

   Opens the default webcam and displays real-time video with filter options.
   Use keyboard controls to switch between different effects.

================================================================================
                              KEYBOARD CONTROLS
================================================================================

   Key     Effect
   ---     ------
   q       Quit the application
   s       Save the current frame to a file
   g       Apply OpenCV greyscale conversion
   h       Apply custom greyscale conversion
   p       Apply sepia tone filter
   b       Apply blur filter
   x       Apply Sobel X edge detection (horizontal edges)
   y       Apply Sobel Y edge detection (vertical edges)
   m       Apply gradient magnitude (combined edge detection)
   l       Apply blur and quantize effect
   c       Apply cartoon effect
   f       Toggle face detection overlay
   d       Apply depth estimation (requires ONNX Runtime)
   e       Apply emboss effect
   v       Apply vignette effect
   n       Apply negative/invert effect
   o       Return to original (no filter)

================================================================================
                               REQUIRED FILES
================================================================================

Source Files:
- src/imgDisplay.cpp      - Image display program
- src/vidDisplay.cpp      - Video display program (or vid.cpp)
- src/filter.cpp          - Filter function implementations
- src/filter.h            - Filter function declarations
- src/faceDetect.cpp      - Face detection implementation
- src/faceDetect.h        - Face detection declarations
- src/Makefile            - Build configuration

Data Files:
- data/haarcascade_frontalface_alt2.xml  - Haar cascade for face detection
- data/depth_anything_v2_vits.onnx       - ONNX model for depth estimation

Downloading the Depth Model:
----------------------------
The depth estimation model (97MB) must be downloaded separately:

   curl -L -o data/depth_anything_v2_vits.onnx \
       "https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/resolve/main/depth_anything_v2_vits.onnx"

Or download manually from:
   https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx

Note: The depth estimation feature ('d' key) will show "Depth model not available"
if this file is not present. All other features will work without it.

================================================================================
                              TIME TRAVEL DAYS
================================================================================

Time travel days used: 0

================================================================================
                              ACKNOWLEDGEMENTS
================================================================================

- OpenCV Documentation: https://docs.opencv.org/
  Reference for image processing functions, video capture, and Haar cascades.

- CS 5330 Course Materials
  Lecture notes and project specifications provided guidance on filter
  implementations and computer vision concepts.

- ONNX Runtime Documentation: https://onnxruntime.ai/docs/
  Reference for neural network inference integration.

================================================================================
