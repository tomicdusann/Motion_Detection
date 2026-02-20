#pragma once
#include <opencv2/opencv.hpp>
#include "constants.h"
#include <omp.h>

// ---------------- EROSION ----------------
inline void erosion(const cv::Mat& src, cv::Mat& dst)
{
    dst.setTo(255);

    const int rows = src.rows;
    const int cols = src.cols;

    for (int y = K; y < rows - K; y++) {

        uchar* dstRow = dst.ptr<uchar>(y);

        for (int x = K; x < cols - K; x++) {
            uchar mn = 255;

            for (int dy = -K; dy <= K; dy++) {
                const uchar* srcRow = src.ptr<uchar>(y + dy);

                for (int dx = -K; dx <= K; dx++) {
                    mn = std::min(mn, srcRow[x + dx]);
                }
            }

            dstRow[x] = mn;
        }
    }
}

// ---------------- DILATION ----------------
inline void dilation(const cv::Mat& src, cv::Mat& dst)
{
    dst.setTo(0);

    const int rows = src.rows;
    const int cols = src.cols;

    for (int y = K; y < rows - K; y++) {

        uchar* dstRow = dst.ptr<uchar>(y);

        for (int x = K; x < cols - K; x++) {
            uchar mx = 0;

            for (int dy = -K; dy <= K; dy++) {
                const uchar* srcRow = src.ptr<uchar>(y + dy);

                for (int dx = -K; dx <= K; dx++) {
                    mx = std::max(mx, srcRow[x + dx]);
                }
            }

            dstRow[x] = mx;
        }
    }
}