/*
 * DA2Network.hpp
 *
 * Author: Joseph Defendre
 * Date: January 5, 2026
 *
 * Purpose: Wrapper class for Depth Anything V2 neural network using ONNX Runtime.
 *          Provides an interface to run depth estimation inference on images,
 *          taking BGR images as input and returning normalized depth maps.
 */

#ifndef DA2NETWORK_HPP
#define DA2NETWORK_HPP

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>

class DA2Network {
private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memoryInfo_;

    // Model input/output names (stored as strings to manage lifetime)
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamePtrs_;
    std::vector<const char*> outputNamePtrs_;

    // Model input dimensions
    int64_t inputHeight_;
    int64_t inputWidth_;
    int64_t inputChannels_;

    /**
     * @brief Preprocesses an input BGR image for the model
     * @param input Input BGR image (cv::Mat)
     * @return Preprocessed float vector in NCHW format with RGB channels normalized to [0,1]
     */
    std::vector<float> preprocess(const cv::Mat& input) const {
        cv::Mat resized;
        cv::resize(input, resized, cv::Size(static_cast<int>(inputWidth_),
                                            static_cast<int>(inputHeight_)));

        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // Convert to float and normalize to [0, 1]
        cv::Mat floatImg;
        rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

        // Create output vector in NCHW format
        std::vector<float> inputTensorValues(inputChannels_ * inputHeight_ * inputWidth_);

        // Split channels and arrange in NCHW format
        std::vector<cv::Mat> channels(3);
        cv::split(floatImg, channels);

        size_t channelSize = static_cast<size_t>(inputHeight_ * inputWidth_);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(inputTensorValues.data() + c * channelSize,
                       channels[c].data,
                       channelSize * sizeof(float));
        }

        return inputTensorValues;
    }

    /**
     * @brief Post-processes the model output to create a depth map
     * @param outputData Raw output data from the model
     * @param outputShape Shape of the output tensor
     * @param originalSize Original input image size for resizing
     * @return Normalized depth map as CV_8UC1
     */
    cv::Mat postprocess(const float* outputData,
                        const std::vector<int64_t>& outputShape,
                        const cv::Size& originalSize) const {
        // Determine output dimensions
        int64_t outHeight, outWidth;

        if (outputShape.size() == 4) {
            // NCHW or NHWC format
            outHeight = outputShape[2];
            outWidth = outputShape[3];
        } else if (outputShape.size() == 3) {
            // NHW format
            outHeight = outputShape[1];
            outWidth = outputShape[2];
        } else if (outputShape.size() == 2) {
            // HW format
            outHeight = outputShape[0];
            outWidth = outputShape[1];
        } else {
            throw std::runtime_error("Unexpected output tensor shape");
        }

        size_t totalElements = static_cast<size_t>(outHeight * outWidth);

        // Find min and max for normalization
        float minVal = *std::min_element(outputData, outputData + totalElements);
        float maxVal = *std::max_element(outputData, outputData + totalElements);

        // Avoid division by zero
        float range = maxVal - minVal;
        if (range < 1e-6f) {
            range = 1.0f;
        }

        // Create depth map
        cv::Mat depthMap(static_cast<int>(outHeight), static_cast<int>(outWidth), CV_32FC1);

        for (int y = 0; y < outHeight; ++y) {
            for (int x = 0; x < outWidth; ++x) {
                float val = outputData[y * outWidth + x];
                // Normalize to [0, 1]
                float normalized = (val - minVal) / range;
                depthMap.at<float>(y, x) = normalized;
            }
        }

        // Convert to 8-bit
        cv::Mat depthMap8U;
        depthMap.convertTo(depthMap8U, CV_8UC1, 255.0);

        // Resize to original input size
        cv::Mat resizedDepth;
        cv::resize(depthMap8U, resizedDepth, originalSize, 0, 0, cv::INTER_LINEAR);

        return resizedDepth;
    }

public:
    /**
     * @brief Constructs a DA2Network with the specified ONNX model
     * @param modelPath Path to the Depth Anything V2 ONNX model file
     * @throws std::runtime_error if model loading fails
     */
    explicit DA2Network(const std::string& modelPath)
        : env_(ORT_LOGGING_LEVEL_WARNING, "DA2Network"),
          session_(nullptr),
          memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          inputHeight_(518),
          inputWidth_(518),
          inputChannels_(3) {

        // Configure session options
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create session
        try {
            session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("Failed to load ONNX model: ") + e.what());
        }

        // Get allocator for retrieving input/output names
        Ort::AllocatorWithDefaultOptions allocator;

        // Get input information
        size_t numInputNodes = session_.GetInputCount();
        inputNames_.reserve(numInputNodes);
        inputNamePtrs_.reserve(numInputNodes);

        for (size_t i = 0; i < numInputNodes; ++i) {
            auto inputName = session_.GetInputNameAllocated(i, allocator);
            inputNames_.push_back(inputName.get());

            // Get input shape
            Ort::TypeInfo typeInfo = session_.GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> inputShape = tensorInfo.GetShape();

            // Update dimensions if available (assuming NCHW format)
            if (inputShape.size() >= 4) {
                if (inputShape[1] > 0) inputChannels_ = inputShape[1];
                if (inputShape[2] > 0) inputHeight_ = inputShape[2];
                if (inputShape[3] > 0) inputWidth_ = inputShape[3];
            }
        }

        // Set up input name pointers
        for (const auto& name : inputNames_) {
            inputNamePtrs_.push_back(name.c_str());
        }

        // Get output information
        size_t numOutputNodes = session_.GetOutputCount();
        outputNames_.reserve(numOutputNodes);
        outputNamePtrs_.reserve(numOutputNodes);

        for (size_t i = 0; i < numOutputNodes; ++i) {
            auto outputName = session_.GetOutputNameAllocated(i, allocator);
            outputNames_.push_back(outputName.get());
        }

        // Set up output name pointers
        for (const auto& name : outputNames_) {
            outputNamePtrs_.push_back(name.c_str());
        }
    }

    /**
     * @brief Runs depth estimation inference on an input image
     * @param input Input BGR image (cv::Mat, any size)
     * @return Depth map as CV_8UC1, same size as input, with values 0-255
     *         where higher values typically indicate greater depth
     * @throws std::runtime_error if inference fails
     */
    cv::Mat forward(const cv::Mat& input) {
        if (input.empty()) {
            throw std::runtime_error("Input image is empty");
        }

        if (input.channels() != 3) {
            throw std::runtime_error("Input image must have 3 channels (BGR)");
        }

        // Store original size for later resizing
        cv::Size originalSize = input.size();

        // Preprocess the input image
        std::vector<float> inputTensorValues = preprocess(input);

        // Create input tensor shape (NCHW format)
        std::vector<int64_t> inputShape = {1, inputChannels_, inputHeight_, inputWidth_};

        // Create input tensor
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo_,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size()
        );

        // Run inference
        std::vector<Ort::Value> outputTensors;
        try {
            outputTensors = session_.Run(
                Ort::RunOptions{nullptr},
                inputNamePtrs_.data(),
                &inputTensor,
                1,
                outputNamePtrs_.data(),
                outputNamePtrs_.size()
            );
        } catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("Inference failed: ") + e.what());
        }

        if (outputTensors.empty()) {
            throw std::runtime_error("No output tensors returned from inference");
        }

        // Get output tensor info
        Ort::TensorTypeAndShapeInfo outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputShape = outputInfo.GetShape();

        // Get output data
        const float* outputData = outputTensors[0].GetTensorData<float>();

        // Post-process and return depth map
        return postprocess(outputData, outputShape, originalSize);
    }

    /**
     * @brief Gets the expected input height for the model
     * @return Input height in pixels
     */
    int64_t getInputHeight() const { return inputHeight_; }

    /**
     * @brief Gets the expected input width for the model
     * @return Input width in pixels
     */
    int64_t getInputWidth() const { return inputWidth_; }

    /**
     * @brief Gets the number of input channels expected by the model
     * @return Number of input channels (typically 3 for RGB)
     */
    int64_t getInputChannels() const { return inputChannels_; }
};

#endif // DA2NETWORK_HPP
