#pragma once
#include <opencv2/opencv.hpp>
#include <cmath>

// ---------------- BGR -> GRAY ----------------
inline void bgr_to_gray(const cv::Mat& bgr, cv::Mat& gray)
{
    const int rows = bgr.rows;
    const int cols = bgr.cols;

    for (int y = 0; y < rows; y++) {        
        const uchar* src = bgr.ptr<uchar>(y);
        uchar* dst = gray.ptr<uchar>(y);    

        for (int x = 0; x < cols; x++) {    
            int idx = 3 * x;
            dst[x] = static_cast<uchar>(
                0.114f * src[idx] +
                0.587f * src[idx + 1] +
                0.299f * src[idx + 2]
                );
        }
    }
}

// ---------------- ABS DIFF + THRESH ----------------
inline void absdiff_thresh(const cv::Mat& cur, const cv::Mat& bg,
    cv::Mat& mask, unsigned char thresh)
{
    const int total = cur.total();
    const uchar* c = cur.data;
    const uchar* b = bg.data;
    uchar* m = mask.data;

    for (int i = 0; i < total; i++) {
        int d = std::abs(int(c[i]) - int(b[i]));
        m[i] = (d > thresh) ? 255 : 0;
    }
}

// ---------------- EDGE THINNING ----------------
inline void edge_thinning(const cv::Mat& mask,
    const cv::Mat& eroded,
    cv::Mat& out)
{
    const int total = mask.total();
    const uchar* m = mask.data;
    const uchar* e = eroded.data;
    uchar* o = out.data;

    for (int i = 0; i < total; i++) {
        o[i] = (m[i] > e[i]) ? 255 : 0;
    }
}

// ---------------- BLUR (SEPARABLE) ----------------
inline void blur_h(const cv::Mat& src, cv::Mat& dst)
{
    const int rows = src.rows;
    const int cols = src.cols;

    for (int y = 0; y < rows; y++) {
        const uchar* s = src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);

        d[0] = 0;
        d[cols - 1] = 0;

        for (int x = 1; x < cols - 1; x++) {
            d[x] = (s[x - 1] + s[x] + s[x + 1]) / 3;
        }
    }
}

inline void blur_v(const cv::Mat& src, cv::Mat& dst)
{
    const int rows = src.rows;
    const int cols = src.cols;

    for (int y = 1; y < rows - 1; y++) {
        const uchar* s0 = src.ptr<uchar>(y - 1);
        const uchar* s1 = src.ptr<uchar>(y);
        const uchar* s2 = src.ptr<uchar>(y + 1);
        uchar* d = dst.ptr<uchar>(y);

        for (int x = 0; x < cols; x++) {
            d[x] = (s0[x] + s1[x] + s2[x]) / 3;
        }
    }

    // ivice
    dst.row(0).setTo(0);
    dst.row(rows - 1).setTo(0);
}

// ---------------- BACKGROUND UPDATE ----------------
inline void update_bg(const cv::Mat& cur, cv::Mat& bg, float alpha)
{
    const int total = cur.total();
    const uchar* c = cur.data;
    uchar* b = bg.data;
    const float inv = 1.0f - alpha;

    for (int i = 0; i < total; i++) {
        b[i] = static_cast<uchar>(
            alpha * c[i] + inv * b[i]
            );
    }
}