/*
 * imgDisplay.cpp
 *
 * Author: Joseph Defendre
 * Date: January 5, 2026
 *
 * Purpose: This program reads an image file from a command line argument
 * (or a default path if none provided), displays it in a window, and
 * allows the user to interact with it via keyboard commands.
 *
 * Keypresses:
 *   'q' - Quit the program
 *   's' - Save the current image to a file
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Default image path if no argument provided
    std::string imagePath = "image.jpg";

    // Check if an image path was provided as a command line argument
    if (argc > 1) {
        imagePath = argv[1];
    } else {
        std::cout << "No image path provided. Using default: " << imagePath << std::endl;
    }

    // Read the image from the specified path
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not read the image file: " << imagePath << std::endl;
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully: " << imagePath << std::endl;
    std::cout << "Image size: " << image.cols << " x " << image.rows << std::endl;
    std::cout << "Press 'q' to quit, 's' to save the image" << std::endl;

    // Create a window to display the image
    const std::string windowName = "Display Image";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Display the image
    cv::imshow(windowName, image);

    // Main loop - wait for keypresses
    while (true) {
        // Wait indefinitely for a keypress (0 means wait forever)
        // Use waitKey(30) for a 30ms delay if you want non-blocking behavior
        int key = cv::waitKey(0);

        // Handle keypress
        if (key == 'q' || key == 'Q') {
            // Quit the program
            std::cout << "Quitting program..." << std::endl;
            break;
        } else if (key == 's' || key == 'S') {
            // Save the current image
            std::string outputPath = "saved_image.png";

            // Attempt to save the image
            bool success = cv::imwrite(outputPath, image);

            if (success) {
                std::cout << "Image saved successfully to: " << outputPath << std::endl;
            } else {
                std::cerr << "Error: Failed to save the image to: " << outputPath << std::endl;
            }
        }
    }

    // Clean up - destroy the window
    cv::destroyAllWindows();

    return 0;
}
