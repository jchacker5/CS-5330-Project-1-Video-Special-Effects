# Makefile for Project 1: Video Special Effects
# Author: Joseph Defendre
# Date: January 2026
# Purpose: Build OpenCV-based image and video processing applications
#          for macOS Apple Silicon (M4)

# Compiler settings
CC = clang++
CFLAGS = -std=c++17 -Wall -O2

# OpenCV configuration (via pkg-config)
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# ONNX Runtime configuration (for face detection neural network)
ONNX_INCLUDE = -I/opt/homebrew/include/onnxruntime
ONNX_LIB = -L/opt/homebrew/lib -lonnxruntime

# Directory configuration
BINDIR = ../bin
INCDIR = ../include

# Combined flags
ALL_CFLAGS = $(CFLAGS) $(OPENCV_CFLAGS) $(ONNX_INCLUDE) -I$(INCDIR)
LDFLAGS = $(OPENCV_LIBS) $(ONNX_LIB)

# Header dependencies
HEADERS = $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp)

# =============================================================================
# Targets
# =============================================================================

.PHONY: all clean directories

all: directories $(BINDIR)/imgDisplay $(BINDIR)/vid

# Create bin directory if it doesn't exist
directories:
	@mkdir -p $(BINDIR)

# -----------------------------------------------------------------------------
# Image Display Application
# -----------------------------------------------------------------------------
$(BINDIR)/imgDisplay: imgDisplay.o
	$(CC) $^ -o $@ $(LDFLAGS)

imgDisplay.o: imgDisplay.cpp $(HEADERS)
	$(CC) $(ALL_CFLAGS) -c $< -o $@

# -----------------------------------------------------------------------------
# Video Display Application (with all effects)
# -----------------------------------------------------------------------------
$(BINDIR)/vid: vidDisplay.o filters.o faceDetect.o
	$(CC) $^ -o $@ $(LDFLAGS)

vidDisplay.o: vidDisplay.cpp $(HEADERS)
	$(CC) $(ALL_CFLAGS) -c $< -o $@

filters.o: filters.cpp $(HEADERS)
	$(CC) $(ALL_CFLAGS) -c $< -o $@

faceDetect.o: faceDetect.cpp $(HEADERS)
	$(CC) $(ALL_CFLAGS) -c $< -o $@

# -----------------------------------------------------------------------------
# Pattern rule for generic .cpp to .o compilation
# -----------------------------------------------------------------------------
%.o: %.cpp $(HEADERS)
	$(CC) $(ALL_CFLAGS) -c $< -o $@

# -----------------------------------------------------------------------------
# Clean up build artifacts
# -----------------------------------------------------------------------------
clean:
	rm -f *.o
	rm -f $(BINDIR)/imgDisplay $(BINDIR)/vid

# =============================================================================
# Notes for Apple Silicon (M4) with Homebrew
# =============================================================================
#
# If you encounter runtime linking issues with ONNX Runtime, you may need to
# add the Homebrew library path to the executable's rpath:
#
#   install_name_tool -add_rpath /opt/homebrew/lib $(BINDIR)/vid
#
# Alternatively, set the environment variable before running:
#
#   export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
#
# To verify OpenCV installation:
#   pkg-config --modversion opencv4
#   pkg-config --cflags --libs opencv4
#
# =============================================================================
