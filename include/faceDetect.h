/*
 * Joseph Defendre
 * January 5, 2026
 * Header file for face detection functions using OpenCV's Haar Cascade Classifier
 */

#ifndef FACEDETECT_H
#define FACEDETECT_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Detects faces in a grayscale image using Haar Cascade Classifier
 * @param grey Input grayscale image
 * @param faces Output vector of rectangles containing detected faces
 * @return 0 on success, -1 on failure
 */
int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces);

/**
 * Draws rectangles around detected faces on the frame
 * @param frame Input/output color image to draw boxes on
 * @param faces Vector of rectangles representing detected faces
 * @param minWidth Minimum width of face to draw (default: 50)
 * @param scale Scale factor for the rectangles (default: 1.0)
 * @return 0 on success, -1 on failure
 */
int drawBoxes(cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth = 50, float scale = 1.0);

#endif // FACEDETECT_H
