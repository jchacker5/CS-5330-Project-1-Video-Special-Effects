/*
 * vidDisplay.cpp
 *
 * Author: Joseph Defendre
 * Date: January 5, 2026
 *
 * Purpose: Main video capture and effects program for real-time video processing.
 *          Captures video from camera and applies various image processing filters
 *          and effects based on user keyboard input. Supports grayscale conversion,
 *          blur, edge detection, cartoon effects, face detection, depth estimation,
 *          and other artistic filters.
 */

#include <opencv2/opencv.hpp>
#include "../include/filters.h"
#include "../include/faceDetect.h"
#include "../include/DA2Network.hpp"
#include <iostream>
#include <chrono>
#include <memory>
#include <fstream>

// Path to the Depth Anything V2 ONNX model
const std::string DA2_MODEL_PATH = "../data/depth_anything_v2_vits.onnx";

/**
 * Generates a timestamp string for file naming
 * @return Timestamp string in format YYYYMMDD_HHMMSS
 */
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&time);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", tm);
    return std::string(buffer);
}

/**
 * Checks if a file exists at the given path
 * @param path Path to check
 * @return true if file exists, false otherwise
 */
bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

/**
 * Applies a negative/inverted color effect to an image
 * @param src Input image (CV_8UC3)
 * @param dst Output inverted image (CV_8UC3)
 */
void invertColors(const cv::Mat& src, cv::Mat& dst) {
    dst = cv::Scalar(255, 255, 255) - src;
}

/**
 * Displays the current mode on the frame
 * @param frame Frame to display text on
 * @param mode Current mode character
 */
void displayModeText(cv::Mat& frame, char mode) {
    std::string modeText;
    switch (mode) {
        case 'o': modeText = "Original"; break;
        case 'g': modeText = "Greyscale (OpenCV)"; break;
        case 'h': modeText = "Greyscale (Custom)"; break;
        case 'p': modeText = "Sepia"; break;
        case 'b': modeText = "Blur 5x5"; break;
        case 'x': modeText = "Sobel X"; break;
        case 'y': modeText = "Sobel Y"; break;
        case 'm': modeText = "Gradient Magnitude"; break;
        case 'l': modeText = "Blur + Quantize"; break;
        case 'c': modeText = "Cartoon"; break;
        case 'f': modeText = "Face Detection"; break;
        case 'd': modeText = "Depth Estimation"; break;
        case 'e': modeText = "Emboss"; break;
        case 'v': modeText = "Vignette"; break;
        case 'n': modeText = "Negative"; break;
        default: modeText = "Unknown"; break;
    }

    cv::putText(frame, modeText, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
}

int main(int argc, char* argv[]) {
    // Open video capture device (default camera at index 0)
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video capture device." << std::endl;
        return -1;
    }

    // Set camera properties for better performance
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::cout << "Video capture initialized successfully." << std::endl;
    std::cout << "Resolution: " << cap.get(cv::CAP_PROP_FRAME_WIDTH)
              << "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    // Create display window
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);

    // Initialize DA2Network for depth estimation (lazy initialization)
    std::unique_ptr<DA2Network> depthNetwork = nullptr;
    bool depthNetworkInitialized = false;
    bool depthNetworkFailed = false;

    // Current mode (default: original)
    char mode = 'o';

    // Frame buffers
    cv::Mat frame, display;
    cv::Mat grey, greyBGR;
    cv::Mat sobelX, sobelY, magnitude_img;
    cv::Mat depthMap, depthColorized;
    std::vector<cv::Rect> faces;

    std::cout << "\n=== Video Effects Controls ===" << std::endl;
    std::cout << "q - Quit" << std::endl;
    std::cout << "s - Save current frame" << std::endl;
    std::cout << "o - Original (no filter)" << std::endl;
    std::cout << "g - Greyscale (OpenCV)" << std::endl;
    std::cout << "h - Greyscale (Custom)" << std::endl;
    std::cout << "p - Sepia tone" << std::endl;
    std::cout << "b - Blur (5x5)" << std::endl;
    std::cout << "x - Sobel X" << std::endl;
    std::cout << "y - Sobel Y" << std::endl;
    std::cout << "m - Gradient magnitude" << std::endl;
    std::cout << "l - Blur + Quantize" << std::endl;
    std::cout << "c - Cartoon" << std::endl;
    std::cout << "f - Face detection" << std::endl;
    std::cout << "d - Depth estimation" << std::endl;
    std::cout << "e - Emboss" << std::endl;
    std::cout << "v - Vignette" << std::endl;
    std::cout << "n - Negative/Inverted" << std::endl;
    std::cout << "==============================\n" << std::endl;

    // Main processing loop
    while (true) {
        // Capture frame
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured." << std::endl;
            break;
        }

        // Apply filter based on current mode
        switch (mode) {
            case 'o':  // Original - no filter
                display = frame.clone();
                break;

            case 'g':  // Greyscale using OpenCV
                cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
                cv::cvtColor(grey, display, cv::COLOR_GRAY2BGR);
                break;

            case 'h':  // Custom greyscale (returns 3-channel BGR)
                greyscale(frame, display);
                break;

            case 'p':  // Sepia tone
                sepia(frame, display);
                break;

            case 'b':  // Blur using optimized 5x5 filter
                blur5x5_2(frame, display);
                break;

            case 'x':  // Sobel X
                sobelX3x3(frame, sobelX);
                cv::convertScaleAbs(sobelX, display);
                break;

            case 'y':  // Sobel Y
                sobelY3x3(frame, sobelY);
                cv::convertScaleAbs(sobelY, display);
                break;

            case 'm':  // Gradient magnitude
                sobelX3x3(frame, sobelX);
                sobelY3x3(frame, sobelY);
                magnitude(sobelX, sobelY, display);
                break;

            case 'l':  // Blur and quantize (10 levels)
                blurQuantize(frame, display, 10);
                break;

            case 'c':  // Cartoon effect
                cartoon(frame, display, 10, 15);
                break;

            case 'f':  // Face detection
                display = frame.clone();
                cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
                faces.clear();
                if (detectFaces(grey, faces) == 0) {
                    drawBoxes(display, faces);
                }
                break;

            case 'd':  // Depth estimation
                // Lazy initialize depth network
                if (!depthNetworkInitialized && !depthNetworkFailed) {
                    if (fileExists(DA2_MODEL_PATH)) {
                        try {
                            std::cout << "Loading depth estimation model..." << std::endl;
                            depthNetwork = std::make_unique<DA2Network>(DA2_MODEL_PATH);
                            depthNetworkInitialized = true;
                            std::cout << "Depth estimation model loaded successfully." << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Failed to load depth model: " << e.what() << std::endl;
                            depthNetworkFailed = true;
                        }
                    } else {
                        std::cerr << "Depth model not found at: " << DA2_MODEL_PATH << std::endl;
                        std::cerr << "Depth estimation will not be available." << std::endl;
                        depthNetworkFailed = true;
                    }
                }

                if (depthNetworkInitialized && depthNetwork) {
                    try {
                        depthMap = depthNetwork->forward(frame);
                        cv::applyColorMap(depthMap, display, cv::COLORMAP_INFERNO);
                    } catch (const std::exception& e) {
                        std::cerr << "Depth inference error: " << e.what() << std::endl;
                        display = frame.clone();
                        cv::putText(display, "Depth Error", cv::Point(10, 60),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                    }
                } else {
                    display = frame.clone();
                    cv::putText(display, "Depth model not available", cv::Point(10, 60),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                }
                break;

            case 'e':  // Emboss effect
                emboss(frame, display);
                break;

            case 'v':  // Vignette effect
                vignette(frame, display, 0.5f);
                break;

            case 'n':  // Negative/Inverted colors
                invertColors(frame, display);
                break;

            default:
                display = frame.clone();
                break;
        }

        // Display current mode text on frame
        displayModeText(display, mode);

        // Show the frame
        cv::imshow("Video", display);

        // Handle keyboard input
        int key = cv::waitKey(10) & 0xFF;

        if (key == 'q') {
            // Quit
            std::cout << "Quitting..." << std::endl;
            break;
        } else if (key == 's') {
            // Save current frame with timestamp
            std::string filename = "capture_" + getTimestamp() + ".png";
            if (cv::imwrite(filename, display)) {
                std::cout << "Frame saved to: " << filename << std::endl;
            } else {
                std::cerr << "Error: Failed to save frame." << std::endl;
            }
        } else if (key == 'o' || key == 'g' || key == 'h' || key == 'p' ||
                   key == 'b' || key == 'x' || key == 'y' || key == 'm' ||
                   key == 'l' || key == 'c' || key == 'f' || key == 'd' ||
                   key == 'e' || key == 'v' || key == 'n') {
            // Update mode
            mode = static_cast<char>(key);
            std::cout << "Mode changed to: " << mode << std::endl;
        }
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Video display terminated successfully." << std::endl;
    return 0;
}
