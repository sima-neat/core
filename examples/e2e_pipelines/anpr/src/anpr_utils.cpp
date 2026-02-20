#include "anpr_utils.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <map>
#include <iostream>

namespace anpr {

const YUVColor COLOR_DEFAULT   = {255, 128, 128}; // White-ish
const YUVColor COLOR_LP        = {255, 128, 128};
const YUVColor COLOR_HLP       = {225, 0, 148};   // Yellow-ish
const YUVColor COLOR_2W        = {178, 166, 16};  // Cyan-ish
const YUVColor COLOR_3W        = {105, 202, 221}; // Magenta/Purple-ish
const YUVColor COLOR_HELMET    = {149, 43, 21};   // Green-ish
const YUVColor COLOR_NO_HELMET = {76, 84, 255};   // Red-ish
const YUVColor COLOR_LMV       = {80, 240, 110};  // Blue-ish
const YUVColor COLOR_HMV       = {179, 70, 190};  // Orange-ish

static std::map<std::string, YUVColor> LABEL_COLORS;

void init_colors() {
    if (!LABEL_COLORS.empty()) return;
    LABEL_COLORS["default"] = COLOR_DEFAULT;
    LABEL_COLORS["lp"] = COLOR_LP;
    LABEL_COLORS["hlp"] = COLOR_HLP;
    LABEL_COLORS["2w"] = COLOR_2W;
    LABEL_COLORS["3w"] = COLOR_3W;
    LABEL_COLORS["helmet"] = COLOR_HELMET;
    LABEL_COLORS["no_helmet"] = COLOR_NO_HELMET;
    LABEL_COLORS["lmv"] = COLOR_LMV;
    LABEL_COLORS["hmv"] = COLOR_HMV;
}

YUVColor get_color_for_label(const std::string& label) {
    init_colors();
    auto it = LABEL_COLORS.find(label);
    if (it != LABEL_COLORS.end()) {
        return it->second;
    }
    return COLOR_DEFAULT;
}

void draw_nv12_box(uint8_t* y_plane, uint8_t* uv_plane, int width, int height,
                   int x1, int y1, int x2, int y2, YUVColor color, int thickness) {
    // Clip coordinates
    x1 = std::max(0, std::min(x1, width));
    y1 = std::max(0, std::min(y1, height));
    x2 = std::max(0, std::min(x2, width));
    y2 = std::max(0, std::min(y2, height));

    if (x1 >= x2 || y1 >= y2) return;

    // Helper to draw horizontal line on Y
    auto draw_h_line_y = [&](int y, int start_x, int end_x) {
        if (y < 0 || y >= height) return;
        std::fill(y_plane + y * width + start_x, y_plane + y * width + end_x, color.y);
    };

    // Helper to draw vertical line on Y
    auto draw_v_line_y = [&](int x, int start_y, int end_y) {
        if (x < 0 || x >= width) return;
        for (int y = start_y; y < end_y; ++y) {
            y_plane[y * width + x] = color.y;
        }
    };
    
    // Draw Y plane box
    for (int t = 0; t < thickness; ++t) {
        draw_h_line_y(y1 + t, x1, x2);         // Top
        draw_h_line_y(y2 - 1 - t, x1, x2);     // Bottom
        draw_v_line_y(x1 + t, y1, y2);         // Left
        draw_v_line_y(x2 - 1 - t, y1, y2);     // Right
    }

    // Draw UV plane box (subsampled)
    int uv_width = width;
    int uv_height = height / 2;
    int uv_y1 = y1 / 2;
    int uv_y2 = y2 / 2;
    int uv_x1 = x1 - (x1 % 2);
    int uv_x2 = x2 - (x2 % 2);
    int uv_th = std::max(1, thickness / 2);

    auto set_uv_pixel = [&](int __y, int __x) {
        if (__y < 0 || __y >= uv_height || __x < 0 || __x >= uv_width) return;
        int idx = __y * uv_width + __x; // Stride is width (interleaved)
        // Ensure even alignment
        idx = idx - (idx % 2);
        uv_plane[idx] = color.u;
        uv_plane[idx + 1] = color.v;
    };
    
    auto draw_h_line_uv = [&](int y, int start_x, int end_x) {
         for (int x = start_x; x < end_x; x += 2) { // Step by 2 for UV pairs
             set_uv_pixel(y, x);
         }
    };
    
    auto draw_v_line_uv = [&](int x, int start_y, int end_y) {
        for (int y = start_y; y < end_y; ++y) {
            set_uv_pixel(y, x);
        }
    };

    for (int t = 0; t < uv_th; ++t) {
        draw_h_line_uv(uv_y1 + t, uv_x1, uv_x2);
        draw_h_line_uv(uv_y2 - 1 - t, uv_x1, uv_x2);
        draw_v_line_uv(uv_x1 + (2*t), uv_y1, uv_y2); // Horizontal step is 2 bytes
        draw_v_line_uv(uv_x2 - 2 - (2*t), uv_y1, uv_y2);
    }
}

void draw_nv12_text(uint8_t* y_plane, int width, int height,
                    const std::string& text, int x, int y, uint8_t color_val) {
    if (text.empty()) return;
    
    // We can wrap the Y plane in a cv::Mat for convenient text drawing
    // Y plane is just a grayscale image of size width x height
    cv::Mat y_mat(height, width, CV_8UC1, y_plane);
    
    // Use OpenCV to draw text. Note: this modifies the buffer directly.
    cv::putText(y_mat, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(color_val), 1, cv::LINE_AA);
}

cv::Mat crop_nv12_to_bgr(const uint8_t* y_plane, const uint8_t* uv_plane, int width, int height,
                         int x1, int y1, int x2, int y2) {
    // 1. Align coordinates to even
    x1 = x1 & ~1; y1 = y1 & ~1;
    x2 = x2 & ~1; y2 = y2 & ~1;
    
    x1 = std::max(0, x1); y1 = std::max(0, y1);
    x2 = std::min(width, x2); y2 = std::min(height, y2);
    
    if (x2 <= x1 || y2 <= y1) return cv::Mat();
    
    int crop_w = x2 - x1;
    int crop_h = y2 - y1;
    
    // 2. Extract Y
    cv::Mat y_full(height, width, CV_8UC1, const_cast<uint8_t*>(y_plane));
    cv::Mat y_crop = y_full(cv::Rect(x1, y1, crop_w, crop_h)).clone();
    
    // 3. Extract UV
    // UV plane is height/2 x width (interleaved)
    cv::Mat uv_full(height / 2, width, CV_8UC1, const_cast<uint8_t*>(uv_plane));
    cv::Mat uv_crop = uv_full(cv::Rect(x1, y1/2, crop_w, crop_h/2)).clone();
    
    // 4. Construct NV12 Mat
    // NV12 in OpenCV is typically height*1.5 rows
    cv::Mat nv12_crop(crop_h + crop_h/2, crop_w, CV_8UC1);
    y_crop.copyTo(nv12_crop(cv::Rect(0, 0, crop_w, crop_h)));
    uv_crop.copyTo(nv12_crop(cv::Rect(0, crop_h, crop_w, crop_h/2)));
    
    // 5. Convert to BGR
    cv::Mat bgr;
    cv::cvtColor(nv12_crop, bgr, cv::COLOR_YUV2BGR_NV12);
    
    return bgr;
}

cv::Mat preprocess_crop_for_ocr(const cv::Mat& crop_bgr, int target_size) {
    if (crop_bgr.empty()) return cv::Mat();
    
    // Convert to RGB
    cv::Mat crop_rgb;
    cv::cvtColor(crop_bgr, crop_rgb, cv::COLOR_BGR2RGB);
    
    int h = crop_rgb.rows;
    int w = crop_rgb.cols;
    
    // Resize preserving aspect ratio
    double scale = (double)target_size / std::max(h, w);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);
    
    int type = (scale < 1) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::Mat resized;
    cv::resize(crop_rgb, resized, cv::Size(new_w, new_h), 0, 0, type);
    
    // Pad to square
    cv::Mat canvas = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    int top = (target_size - new_h) / 2;
    int left = (target_size - new_w) / 2;
    
    resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
    
    return canvas;
}

} // namespace anpr
