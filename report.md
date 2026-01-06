# CS 5330 - Pattern Recognition and Computer Vision
# Project 1: Video Special Effects Report

**Name:** Joseph Defendre
**Date:** January 2026
**Course:** CS 5330 Pattern Recognition and Computer Vision

---

## 1. Project Overview

This project implements a real-time video processing application using OpenCV that applies various image filters and special effects to live video from a webcam. The application demonstrates fundamental computer vision techniques including color space manipulation, convolution-based filtering, edge detection, face detection, and neural network-based depth estimation.

---

## 2. Implementation Details

### Task 1: Image Display (imgDisplay.cpp)

Implemented a basic image display program that reads an image from the command line and displays it in a window. The program supports:
- **'q' key:** Quit the application
- **'s' key:** Save the displayed image with a timestamp

The implementation uses `cv::imread()` for reading and `cv::imshow()` for display, with `cv::waitKey()` for keyboard input handling.

### Task 2-3: Video Display with Save Functionality (vidDisplay.cpp)

Created a live video capture application using `cv::VideoCapture` that opens the default camera and displays frames in real-time. The application:
- Captures frames at 640x480 resolution
- Displays the current filter mode as text overlay
- Saves frames with timestamp filenames when 's' is pressed

### Task 4: OpenCV Greyscale Conversion

Implemented greyscale conversion using OpenCV's built-in `cv::cvtColor()` function with `cv::COLOR_BGR2GRAY`. The result is converted back to BGR for consistent display.

**Key Code:**
```cpp
cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
cv::cvtColor(grey, display, cv::COLOR_GRAY2BGR);
```

### Task 5: Custom Greyscale Implementation

Implemented a custom greyscale function that uses the maximum of the RGB channels rather than the standard weighted average. This approach preserves more brightness information from the brightest channel.

**Algorithm:**
```cpp
grey_value = max(B, max(G, R))
```

This produces a slightly different aesthetic compared to the luminance-weighted standard, often appearing brighter.

### Task 6: Sepia Tone Filter

Implemented a sepia tone filter using matrix transformation on each pixel's BGR values:

**Transformation Matrix:**
```
new_B = 0.272*R + 0.534*G + 0.131*B
new_G = 0.349*R + 0.686*G + 0.168*B
new_R = 0.393*R + 0.769*G + 0.189*B
```

Values are clamped to 255 to prevent overflow.

### Task 7: Naive 5x5 Blur (blur5x5_1)

Implemented a naive 5x5 Gaussian blur using nested loops and the `cv::Mat::at<>()` method for pixel access. Uses a [1,2,4,2,1] kernel normalized by 100.

**Characteristics:**
- Simple, straightforward implementation
- Slower due to non-contiguous memory access
- Each pixel requires 25 multiply-add operations

### Task 8: Optimized Separable 5x5 Blur (blur5x5_2)

Implemented an optimized version using separable filters with row pointer access:

1. **Horizontal pass:** Apply 1x5 [1,2,4,2,1] filter
2. **Vertical pass:** Apply 5x1 [1,2,4,2,1] filter

**Optimizations:**
- Uses `cv::Mat::ptr<>()` for contiguous memory access
- Separable convolution reduces operations from 25 to 10 per pixel
- Better cache utilization

### Timing Comparison Results

Tested on `cathedral.jpeg` image, averaged over 10 iterations:

| Implementation | Time per Image | Speedup |
|---------------|----------------|---------|
| blur5x5_1 (naive) | 0.0121 seconds | 1.0x (baseline) |
| blur5x5_2 (optimized) | 0.0010 seconds | **12.1x faster** |

The optimized separable filter is approximately **12 times faster** than the naive implementation, demonstrating the significant performance benefits of:
1. Separable filter decomposition (25 ops → 10 ops per pixel)
2. Row pointer access vs. at<> method
3. Better memory access patterns

### Task 9: Sobel X (3x3)

Implemented horizontal edge detection using a separable Sobel filter:
- Horizontal: [-1, 0, 1]
- Vertical: [1, 2, 1]

Output is stored as CV_16SC3 (signed 16-bit) to preserve negative gradient values.

### Task 10: Sobel Y (3x3)

Implemented vertical edge detection using a separable Sobel filter:
- Horizontal: [1, 2, 1]
- Vertical: [-1, 0, 1]

Output is stored as CV_16SC3 to preserve negative gradient values.

### Task 11: Gradient Magnitude

Computes the Euclidean magnitude of the Sobel X and Y gradients:

```cpp
magnitude = sqrt(sobelX^2 + sobelY^2)
```

This produces a comprehensive edge map showing edges in all orientations.

### Task 12: Blur and Quantize

Combines blur5x5_2 with color quantization:
1. Apply 5x5 Gaussian blur to reduce noise
2. Quantize each color channel to N levels (default: 10)

**Quantization Formula:**
```cpp
quantized = (value / bucket_size) * bucket_size + bucket_size / 2
```

### Task 13: Cartoon Effect

Creates a cartoon/comic book effect by combining:
1. **Blur and quantize** the image for flat color regions
2. **Edge detection** using gradient magnitude
3. **Threshold edges** to create black outlines
4. **Composite** by darkening quantized image where edges exist

The result resembles hand-drawn animation with distinct outlines and posterized colors.

### Task 14: Face Detection

Integrated OpenCV's Haar Cascade Classifier for real-time face detection:
- Uses `haarcascade_frontalface_alt2.xml` cascade file
- Detects faces using `detectMultiScale()` with scale factor 1.1
- Draws green rectangles around detected faces

The cascade classifier searches multiple locations at different scales, providing robust face detection for frontal faces.

### Task 15: Depth Estimation (Depth Anything V2)

Integrated the Depth Anything V2 neural network for monocular depth estimation:
- Uses ONNX Runtime for model inference
- Loads `depth_anything_v2_vits.onnx` (ViT-Small model, 97MB)
- Preprocesses frames to 518x518 with ImageNet normalization
- Outputs depth map visualized with COLORMAP_INFERNO

The network estimates relative depth from a single RGB image, with warmer colors indicating closer objects.

---

## 3. Additional Effects (Extensions)

### Effect 1: Emboss

Creates a 3D relief effect by computing directional gradients:
- Uses Sobel X and Y gradients
- Combines with direction vector (0.7071, 0.7071) for 45-degree lighting
- Adds 128 offset for neutral gray background
- Creates appearance of raised/lowered surface features

### Effect 2: Vignette

Applies a classic camera vignette effect:
- Creates Gaussian falloff mask from image center
- Darkens corners while preserving center brightness
- Adjustable strength parameter (default: 0.5)
- Simulates optical lens characteristics

### Effect 3: Negative/Invert

Inverts all color values to create a photographic negative:
```cpp
output = 255 - input
```

Useful for artistic effects and image analysis.

---

## 4. Program Controls

| Key | Effect |
|-----|--------|
| q | Quit application |
| s | Save current frame |
| o | Original (no filter) |
| g | Greyscale (OpenCV) |
| h | Greyscale (Custom) |
| p | Sepia tone |
| b | Blur 5x5 |
| x | Sobel X |
| y | Sobel Y |
| m | Gradient magnitude |
| l | Blur + Quantize |
| c | Cartoon |
| f | Face detection |
| d | Depth estimation |
| e | Emboss |
| v | Vignette |
| n | Negative |

---

## 5. Build Instructions

### Prerequisites
- OpenCV 4.x
- ONNX Runtime (for depth estimation)
- C++17 compatible compiler

### Build Commands
```bash
cd src
make all
```

### Download Depth Model
```bash
curl -L -o data/depth_anything_v2_vits.onnx \
    "https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/resolve/main/depth_anything_v2_vits.onnx"
```

---

## 6. Challenges and Solutions

### Challenge 1: Separable Filter Implementation
Understanding the mathematical decomposition of 2D convolution into two 1D passes required careful analysis. The key insight is that a 5x5 Gaussian kernel can be expressed as the outer product of two 1x5 vectors, reducing computational complexity from O(n^2) to O(2n).

### Challenge 2: Signed Integer Handling for Sobel
The Sobel operators produce negative values for gradients in certain directions. Using CV_16SC3 (signed 16-bit) preserves this information, with proper conversion using `cv::convertScaleAbs()` for display.

### Challenge 3: ONNX Runtime Integration
Integrating the Depth Anything V2 model required:
- Proper input preprocessing (resize, normalize, NCHW format)
- Memory management for ONNX Runtime tensors
- Graceful handling when model file is not present

---

## 7. Time Travel Days Used

**0 days**

---

## 8. References

1. OpenCV Documentation: https://docs.opencv.org/
2. ONNX Runtime Documentation: https://onnxruntime.ai/docs/
3. Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
4. CS 5330 Course Materials and Lecture Notes
