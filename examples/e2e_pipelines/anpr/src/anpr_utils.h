#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

// Forward declarations
namespace simaai::neat {
struct Tensor;
}

namespace anpr {

/**
 * @brief YUV color definition
 */
struct YUVColor {
    uint8_t y;
    uint8_t u;
    uint8_t v;
};

// Predefined colors matching the Python script
extern const YUVColor COLOR_DEFAULT;
extern const YUVColor COLOR_LP;
extern const YUVColor COLOR_HLP;
extern const YUVColor COLOR_2W;
extern const YUVColor COLOR_3W;
extern const YUVColor COLOR_HELMET;
extern const YUVColor COLOR_NO_HELMET;
extern const YUVColor COLOR_LMV;
extern const YUVColor COLOR_HMV;

/**
 * @brief Initialize the static color map
 */
void init_colors();

/**
 * @brief Get color for a class label
 */
YUVColor get_color_for_label(const std::string& label);

/**
 * @brief Draw a colored box on an NV12 buffer (in-place)
 * 
 * @param y_plane Pointer to the Y plane data
 * @param uv_plane Pointer to the UV plane data
 * @param width Image width
 * @param height Image height
 * @param x1 Box left
 * @param y1 Box top
 * @param x2 Box right
 * @param y2 Box bottom
 * @param color YUV color
 * @param thickness Line thickness
 */
void draw_nv12_box(uint8_t* y_plane, uint8_t* uv_plane, int width, int height,
                   int x1, int y1, int x2, int y2, YUVColor color, int thickness = 2);

/**
 * @brief Draw text on NV12 buffer (using OpenCV's putText on Y plane mostly)
 * 
 * @param y_plane Pointer to the Y plane data (treated as grayscale image)
 * @param width Image width
 * @param height Image height
 * @param text Text string
 * @param x X coordinate (bottom-left)
 * @param y Y coordinate (bottom-left)
 * @param color_val Grayscale value (0-255)
 */
void draw_nv12_text(uint8_t* y_plane, int width, int height,
                    const std::string& text, int x, int y, uint8_t color_val = 255);

/**
 * @brief Crop a region from an NV12 buffer and convert it to BGR (OpenCV Mat)
 * 
 * @param y_plane Pointer to the Y plane data
 * @param uv_plane Pointer to the UV plane data
 * @param width Image width
 * @param height Image height
 * @param x1 Crop left
 * @param y1 Crop top
 * @param x2 Crop right
 * @param y2 Crop bottom
 * @return cv::Mat BGR image of the crop, or empty if invalid
 */
cv::Mat crop_nv12_to_bgr(const uint8_t* y_plane, const uint8_t* uv_plane, int width, int height,
                         int x1, int y1, int x2, int y2);

/**
 * @brief Preprocess a BGR crop for OCR (resize to target_size x target_size with padding)
 * 
 * @param crop_bgr Input BGR image
 * @param target_size Target dimension (square)
 * @return cv::Mat RGB image (target_size x target_size) ready for OCR input
 */
cv::Mat preprocess_crop_for_ocr(const cv::Mat& crop_bgr, int target_size);

} // namespace anpr
