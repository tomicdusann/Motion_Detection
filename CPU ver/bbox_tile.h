#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "constants.h"
#include "bbox_merge.h"

inline void find_tile_bboxes(
    const cv::Mat& mask,
    std::vector<BBox>& tiles)
{
    tiles.clear();

    const int rows = mask.rows;
    const int cols = mask.cols;

    const int tiles_x = (cols + TILE - 1) / TILE;
    const int tiles_y = (rows + TILE - 1) / TILE;

    for (int ty = 0; ty < tiles_y; ty++) {
        for (int tx = 0; tx < tiles_x; tx++) {

            int x0 = tx * TILE;
            int y0 = ty * TILE;

            int xmin = cols, ymin = rows;
            int xmax = 0, ymax = 0;
            bool found = false;

            for (int dy = 0; dy < TILE; dy++) {
                for (int dx = 0; dx < TILE; dx++) {
                    int x = x0 + dx;
                    int y = y0 + dy;
                    if (x >= cols || y >= rows) continue;

                    if (mask.at<uchar>(y, x)) {
                        xmin = std::min(xmin, x);
                        ymin = std::min(ymin, y);
                        xmax = std::max(xmax, x);
                        ymax = std::max(ymax, y);
                        found = true;
                    }
                }
            }

            if (found && xmax > xmin && ymax > ymin) {
                tiles.push_back({
                    xmin, ymin,
                    xmax, ymax
                    });
            }
        }
    }
}