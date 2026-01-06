/*
 * filters.h
 *
 * Author: Joseph Defendre
 * Date: January 5, 2026
 *
 * Purpose: Header file containing function declarations for various image
 *          processing filters used in a computer vision video effects project.
 *          Includes grayscale conversion, blur filters, edge detection,
 *          and artistic effects.
 */

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

/*
 * Converts a color image to greyscale using a custom algorithm.
 * This implementation provides an alternative to OpenCV's cvtColor.
 *
 * @param src Input color image (CV_8UC3)
 * @param dst Output greyscale image (CV_8UC1)
 * @return 0 on success, non-zero on failure
 */
int greyscale(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a sepia tone filter to create a warm, vintage appearance.
 * The sepia effect gives images an antique, brownish tint.
 *
 * @param src Input color image (CV_8UC3)
 * @param dst Output sepia-toned image (CV_8UC3)
 * @return 0 on success, non-zero on failure
 */
int sepia(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 5x5 Gaussian blur using a naive implementation.
 * This version applies the 2D kernel directly to each pixel.
 *
 * @param src Input image (CV_8UC3)
 * @param dst Output blurred image (CV_8UC3)
 * @return 0 on success, non-zero on failure
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 5x5 Gaussian blur using separable filters.
 * This optimized version separates the 2D convolution into
 * two 1D convolutions (horizontal and vertical passes).
 *
 * @param src Input image (CV_8UC3)
 * @param dst Output blurred image (CV_8UC3)
 * @return 0 on success, non-zero on failure
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 3x3 Sobel filter in the X direction.
 * Detects vertical edges with positive values on the right.
 *
 * @param src Input image (CV_8UC3)
 * @param dst Output gradient image (CV_16SC3)
 * @return 0 on success, non-zero on failure
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a 3x3 Sobel filter in the Y direction.
 * Detects horizontal edges with positive values pointing up.
 *
 * @param src Input image (CV_8UC3)
 * @param dst Output gradient image (CV_16SC3)
 * @return 0 on success, non-zero on failure
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/*
 * Computes the gradient magnitude from Sobel X and Y outputs.
 * Uses Euclidean distance: magnitude = sqrt(sx^2 + sy^2)
 *
 * @param sx Input Sobel X gradient image (CV_16SC3)
 * @param sy Input Sobel Y gradient image (CV_16SC3)
 * @param dst Output magnitude image (CV_8UC3)
 * @return 0 on success, non-zero on failure
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/*
 * Applies blur followed by color quantization.
 * Reduces the number of colors in the image to create a posterized effect.
 *
 * @param src Input image (CV_8UC3)
 * @param dst Output quantized image (CV_8UC3)
 * @param levels Number of quantization levels per channel
 * @return 0 on success, non-zero on failure
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

/*
 * Creates a cartoon effect by combining edge detection with color quantization.
 * Strong edges are drawn in black over a quantized color image.
 *
 * @param src Input image (CV_8UC3)
 * @param dst Output cartoon image (CV_8UC3)
 * @param levels Number of color quantization levels
 * @param magThreshold Gradient magnitude threshold for edge detection
 * @return 0 on success, non-zero on failure
 */
int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold);

/*
 * Applies an embossing effect to create a 3D raised appearance.
 * Simulates the look of embossed or stamped material.
 *
 * @param src Input image (CV_8UC3)
 * @param dst Output embossed image (CV_8UC3)
 * @return 0 on success, non-zero on failure
 */
int emboss(cv::Mat &src, cv::Mat &dst);

/*
 * Applies a vignette effect that darkens the corners of the image.
 * Creates a spotlight-like focus on the center of the image.
 *
 * @param src Input image (CV_8UC3)
 * @param dst Output vignetted image (CV_8UC3)
 * @param strength Vignette intensity (0.0 = no effect, 1.0 = maximum darkening)
 * @return 0 on success, non-zero on failure
 */
int vignette(cv::Mat &src, cv::Mat &dst, float strength);

#endif /* FILTERS_H */
