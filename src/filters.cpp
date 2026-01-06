/*
 * Joseph Defendre
 * Date: January 5, 2026
 * Purpose: Implementation of image filter functions for computer vision project.
 *          Includes various filters such as greyscale, sepia, blur, edge detection,
 *          cartoon effect, emboss, and vignette.
 */

#include "../include/filters.h"
#include <cmath>
#include <algorithm>

/**
 * Custom greyscale conversion using the maximum RGB channel value.
 * This creates a brighter, more contrasty greyscale than standard methods
 * because it takes the brightest channel rather than averaging.
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output greyscale image stored as BGR (CV_8UC3)
 * @return 0 on success
 */
int greyscale(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Use maximum of RGB channels for a brighter, more dramatic greyscale
            uchar maxVal = std::max({srcRow[j][0], srcRow[j][1], srcRow[j][2]});
            dstRow[j][0] = maxVal; // Blue
            dstRow[j][1] = maxVal; // Green
            dstRow[j][2] = maxVal; // Red
        }
    }

    return 0;
}

/**
 * Sepia tone filter that gives images a warm, vintage appearance.
 * Uses standard sepia transformation matrix with clamping to prevent overflow.
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output sepia-toned image (CV_8UC3)
 * @return 0 on success
 */
int sepia(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Store original BGR values
            float blue = srcRow[j][0];
            float green = srcRow[j][1];
            float red = srcRow[j][2];

            // Apply sepia transformation matrix
            float newBlue = 0.272f * red + 0.534f * green + 0.131f * blue;
            float newGreen = 0.349f * red + 0.686f * green + 0.168f * blue;
            float newRed = 0.393f * red + 0.769f * green + 0.189f * blue;

            // Clamp values to 255
            dstRow[j][0] = static_cast<uchar>(std::min(255.0f, newBlue));
            dstRow[j][1] = static_cast<uchar>(std::min(255.0f, newGreen));
            dstRow[j][2] = static_cast<uchar>(std::min(255.0f, newRed));
        }
    }

    return 0;
}

/**
 * Naive 5x5 Gaussian blur implementation using cv::Mat::at<> method.
 * Uses a 5x5 kernel with weights that approximate a Gaussian distribution.
 * Sum of kernel weights = 100 for easy normalization.
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output blurred image (CV_8UC3)
 * @return 0 on success
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());
    src.copyTo(dst); // Copy border pixels

    // 5x5 Gaussian kernel (sum = 100)
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    // Process pixels avoiding the 2-pixel border
    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Convolve with kernel
            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + ki, j + kj);
                    int weight = kernel[ki + 2][kj + 2];
                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            // Normalize and store
            dst.at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(sumB / 100);
            dst.at<cv::Vec3b>(i, j)[1] = static_cast<uchar>(sumG / 100);
            dst.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(sumR / 100);
        }
    }

    return 0;
}

/**
 * Optimized 5x5 Gaussian blur using separable 1x5 filters and row pointers.
 * The 2D kernel is separated into horizontal and vertical 1D filters [1,2,4,2,1].
 * This reduces operations from O(25n) to O(10n) per pixel.
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output blurred image (CV_8UC3)
 * @return 0 on success
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    // 1D kernel [1, 2, 4, 2, 1] with sum = 10
    int kernel[5] = {1, 2, 4, 2, 1};

    // Temporary image for intermediate result
    cv::Mat temp(src.size(), src.type());
    src.copyTo(temp);
    src.copyTo(dst);

    // Horizontal pass
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tempRow = temp.ptr<cv::Vec3b>(i);

        for (int j = 2; j < src.cols - 2; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int k = -2; k <= 2; k++) {
                int weight = kernel[k + 2];
                sumB += srcRow[j + k][0] * weight;
                sumG += srcRow[j + k][1] * weight;
                sumR += srcRow[j + k][2] * weight;
            }

            tempRow[j][0] = static_cast<uchar>(sumB / 10);
            tempRow[j][1] = static_cast<uchar>(sumG / 10);
            tempRow[j][2] = static_cast<uchar>(sumR / 10);
        }
    }

    // Vertical pass
    for (int i = 2; i < src.rows - 2; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        // Get pointers to the 5 rows we need
        cv::Vec3b *tempRows[5];
        for (int k = -2; k <= 2; k++) {
            tempRows[k + 2] = temp.ptr<cv::Vec3b>(i + k);
        }

        for (int j = 2; j < src.cols - 2; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int k = -2; k <= 2; k++) {
                int weight = kernel[k + 2];
                sumB += tempRows[k + 2][j][0] * weight;
                sumG += tempRows[k + 2][j][1] * weight;
                sumR += tempRows[k + 2][j][2] * weight;
            }

            dstRow[j][0] = static_cast<uchar>(sumB / 10);
            dstRow[j][1] = static_cast<uchar>(sumG / 10);
            dstRow[j][2] = static_cast<uchar>(sumR / 10);
        }
    }

    return 0;
}

/**
 * Sobel X gradient filter using separable 3x3 filters.
 * Horizontal: [-1, 0, 1], Vertical: [1, 2, 1]
 * Detects vertical edges. Positive values indicate edges going right.
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output gradient image (CV_16SC3 - signed short)
 * @return 0 on success
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    // Create output as signed 16-bit
    dst.create(src.size(), CV_16SC3);
    dst.setTo(cv::Scalar(0, 0, 0));

    // Temporary image for intermediate result (signed 16-bit)
    cv::Mat temp(src.size(), CV_16SC3);
    temp.setTo(cv::Scalar(0, 0, 0));

    // Horizontal pass with [-1, 0, 1]
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                tempRow[j][c] = -srcRow[j - 1][c] + srcRow[j + 1][c];
            }
        }
    }

    // Vertical pass with [1, 2, 1]
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3s *tempRowPrev = temp.ptr<cv::Vec3s>(i - 1);
        cv::Vec3s *tempRowCurr = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *tempRowNext = temp.ptr<cv::Vec3s>(i + 1);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = tempRowPrev[j][c] + 2 * tempRowCurr[j][c] + tempRowNext[j][c];
            }
        }
    }

    return 0;
}

/**
 * Sobel Y gradient filter using separable 3x3 filters.
 * Horizontal: [1, 2, 1], Vertical: [-1, 0, 1]
 * Detects horizontal edges. Positive values indicate edges going up.
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output gradient image (CV_16SC3 - signed short)
 * @return 0 on success
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    // Create output as signed 16-bit
    dst.create(src.size(), CV_16SC3);
    dst.setTo(cv::Scalar(0, 0, 0));

    // Temporary image for intermediate result (signed 16-bit)
    cv::Mat temp(src.size(), CV_16SC3);
    temp.setTo(cv::Scalar(0, 0, 0));

    // Horizontal pass with [1, 2, 1]
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                tempRow[j][c] = srcRow[j - 1][c] + 2 * srcRow[j][c] + srcRow[j + 1][c];
            }
        }
    }

    // Vertical pass with [-1, 0, 1] (note: -1 at top, +1 at bottom for "positive up")
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3s *tempRowPrev = temp.ptr<cv::Vec3s>(i - 1);
        cv::Vec3s *tempRowNext = temp.ptr<cv::Vec3s>(i + 1);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                // Positive up means: top row gets positive weight
                dstRow[j][c] = tempRowPrev[j][c] - tempRowNext[j][c];
            }
        }
    }

    return 0;
}

/**
 * Compute gradient magnitude from Sobel X and Y outputs.
 * Uses Euclidean distance: sqrt(sx^2 + sy^2)
 *
 * @param sx Sobel X gradient (CV_16SC3)
 * @param sy Sobel Y gradient (CV_16SC3)
 * @param dst Output magnitude image (CV_8UC3)
 * @return 0 on success
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i < sx.rows; i++) {
        cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                float gx = static_cast<float>(sxRow[j][c]);
                float gy = static_cast<float>(syRow[j][c]);
                float mag = std::sqrt(gx * gx + gy * gy);
                dstRow[j][c] = static_cast<uchar>(std::min(255.0f, mag));
            }
        }
    }

    return 0;
}

/**
 * Blur and quantize filter for creating a stylized effect.
 * First applies Gaussian blur, then reduces the number of color levels.
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output quantized image (CV_8UC3)
 * @param levels Number of quantization levels per channel
 * @return 0 on success
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    // First apply blur
    cv::Mat blurred;
    blur5x5_2(src, blurred);

    dst.create(src.size(), src.type());

    // Calculate bucket size
    float bucket = 255.0f / levels;

    for (int i = 0; i < blurred.rows; i++) {
        cv::Vec3b *blurRow = blurred.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < blurred.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int xt = static_cast<int>(blurRow[j][c] / bucket);
                int xf = static_cast<int>(xt * bucket);
                dstRow[j][c] = static_cast<uchar>(xf);
            }
        }
    }

    return 0;
}

/**
 * Cartoon effect combining blur/quantize with edge detection.
 * Areas with strong gradients (edges) are set to black to create
 * an outline effect on top of the quantized colors.
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output cartoon image (CV_8UC3)
 * @param levels Number of quantization levels
 * @param magThreshold Gradient magnitude threshold for edge detection
 * @return 0 on success
 */
int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold) {
    // Apply blur and quantize
    cv::Mat quantized;
    blurQuantize(src, quantized, levels);

    // Compute gradient magnitude
    cv::Mat sobelx, sobely, mag;
    sobelX3x3(src, sobelx);
    sobelY3x3(src, sobely);
    magnitude(sobelx, sobely, mag);

    dst.create(src.size(), src.type());

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *quantRow = quantized.ptr<cv::Vec3b>(i);
        cv::Vec3b *magRow = mag.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Check if any channel's magnitude exceeds threshold
            int maxMag = std::max({magRow[j][0], magRow[j][1], magRow[j][2]});

            if (maxMag > magThreshold) {
                // Edge pixel - set to black
                dstRow[j][0] = 0;
                dstRow[j][1] = 0;
                dstRow[j][2] = 0;
            } else {
                // Non-edge pixel - use quantized color
                dstRow[j] = quantRow[j];
            }
        }
    }

    return 0;
}

/**
 * Emboss effect using directional gradient.
 * Computes the dot product of the gradient with a light direction (0.7071, 0.7071),
 * which corresponds to light coming from the top-left at 45 degrees.
 * Adds 128 to make the result visible (neutral gray becomes 128).
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output embossed image (CV_8UC3)
 * @return 0 on success
 */
int emboss(cv::Mat &src, cv::Mat &dst) {
    // Compute Sobel gradients
    cv::Mat sobelx, sobely;
    sobelX3x3(src, sobelx);
    sobelY3x3(src, sobely);

    dst.create(src.size(), src.type());

    // Light direction (45 degrees from top-left)
    const float dirX = 0.7071f;
    const float dirY = 0.7071f;

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3s *sxRow = sobelx.ptr<cv::Vec3s>(i);
        cv::Vec3s *syRow = sobely.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                // Dot product with light direction
                float embossVal = sxRow[j][c] * dirX + syRow[j][c] * dirY;

                // Add 128 for visibility and clamp to valid range
                int result = static_cast<int>(embossVal / 4.0f) + 128;
                result = std::max(0, std::min(255, result));
                dstRow[j][c] = static_cast<uchar>(result);
            }
        }
    }

    return 0;
}

/**
 * Vignette effect that darkens the edges of the image.
 * Uses a Gaussian falloff from the center to create a natural-looking
 * darkening effect that draws attention to the center of the image.
 *
 * @param src Input BGR image (CV_8UC3)
 * @param dst Output vignetted image (CV_8UC3)
 * @param strength Vignette strength (0.0 = no effect, higher = stronger darkening)
 * @return 0 on success
 */
int vignette(cv::Mat &src, cv::Mat &dst, float strength) {
    dst.create(src.size(), src.type());

    // Calculate center of image
    float centerX = src.cols / 2.0f;
    float centerY = src.rows / 2.0f;

    // Calculate maximum distance (corner to center)
    float maxDist = std::sqrt(centerX * centerX + centerY * centerY);

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Calculate distance from center (normalized to 0-1)
            float dx = (j - centerX) / maxDist;
            float dy = (i - centerY) / maxDist;
            float dist = std::sqrt(dx * dx + dy * dy);

            // Gaussian falloff
            float falloff = std::exp(-dist * dist * strength * 2.0f);

            // Apply vignette
            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = static_cast<uchar>(srcRow[j][c] * falloff);
            }
        }
    }

    return 0;
}
