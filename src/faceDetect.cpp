/*
 * Joseph Defendre
 * January 5, 2026
 * Implementation of face detection functions using OpenCV's Haar Cascade Classifier
 */

#include "../include/faceDetect.h"
#include <iostream>

// Global Haar Cascade Classifier for face detection
static cv::CascadeClassifier faceCascade;
static bool cascadeLoaded = false;

/**
 * Initializes the cascade classifier by searching multiple locations
 * @return true if cascade loaded successfully, false otherwise
 */
static bool initCascade() {
    if (cascadeLoaded) {
        return true;
    }

    // List of potential paths to search for the cascade file
    std::vector<std::string> cascadePaths = {
        "../data/haarcascade_frontalface_alt2.xml",
        "./haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt2.xml"
    };

    for (const std::string &path : cascadePaths) {
        if (faceCascade.load(path)) {
            std::cout << "Loaded cascade classifier from: " << path << std::endl;
            cascadeLoaded = true;
            return true;
        }
    }

    std::cerr << "Error: Could not load Haar Cascade classifier from any location" << std::endl;
    std::cerr << "Searched locations:" << std::endl;
    for (const std::string &path : cascadePaths) {
        std::cerr << "  - " << path << std::endl;
    }

    return false;
}

/**
 * Detects faces in a grayscale image using Haar Cascade Classifier
 */
int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces) {
    // Initialize cascade if not already done
    if (!initCascade()) {
        return -1;
    }

    // Clear any previous detections
    faces.clear();

    // Check if input image is valid
    if (grey.empty()) {
        std::cerr << "Error: Empty image passed to detectFaces" << std::endl;
        return -1;
    }

    // Detect faces using the cascade classifier
    // Parameters: image, objects, scaleFactor, minNeighbors, flags, minSize, maxSize
    faceCascade.detectMultiScale(
        grey,           // Input grayscale image
        faces,          // Output vector of detected faces
        1.1,            // Scale factor (how much image size is reduced at each scale)
        3,              // Min neighbors (higher = fewer detections but more reliable)
        0,              // Flags (not used in newer OpenCV versions)
        cv::Size(30, 30) // Minimum face size to detect
    );

    return 0;
}

/**
 * Draws rectangles around detected faces on the frame
 */
int drawBoxes(cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth, float scale) {
    // Check if input frame is valid
    if (frame.empty()) {
        std::cerr << "Error: Empty frame passed to drawBoxes" << std::endl;
        return -1;
    }

    // Draw rectangle around each detected face that meets the minimum width requirement
    for (const cv::Rect &face : faces) {
        // Filter faces by minimum width
        if (face.width >= minWidth) {
            // Calculate scaled rectangle if scale is not 1.0
            cv::Rect scaledFace = face;
            if (scale != 1.0f) {
                int newWidth = static_cast<int>(face.width * scale);
                int newHeight = static_cast<int>(face.height * scale);
                int offsetX = (newWidth - face.width) / 2;
                int offsetY = (newHeight - face.height) / 2;

                scaledFace.x = face.x - offsetX;
                scaledFace.y = face.y - offsetY;
                scaledFace.width = newWidth;
                scaledFace.height = newHeight;
            }

            // Draw the rectangle on the frame
            // Using green color with thickness of 2
            cv::rectangle(
                frame,
                scaledFace,
                cv::Scalar(0, 255, 0),  // Green color (BGR format)
                2                        // Line thickness
            );
        }
    }

    return 0;
}
