#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// CPU BBox (ekvivalent CUDA struct-u)
struct BBox {
    int x1, y1, x2, y2;
};

// Check overlap or proximity (8-neighborhood)
inline bool overlap_or_close(
    const BBox& a,
    const BBox& b,
    int gap)
{
    if (a.x2 + gap < b.x1) return false;
    if (b.x2 + gap < a.x1) return false;
    if (a.y2 + gap < b.y1) return false;
    if (b.y2 + gap < a.y1) return false;
    return true;
}

// Merge two bounding boxes
inline BBox merge_bbox(const BBox& a, const BBox& b)
{
    return {
        std::min(a.x1, b.x1),
        std::min(a.y1, b.y1),
        std::max(a.x2, b.x2),
        std::max(a.y2, b.y2)
    };
}

// Merge tile bboxes into object bboxes
inline void merge_tile_bboxes(
    const std::vector<BBox>& tiles,
    std::vector<BBox>& merged,
    int gap)
{
    merged.clear();

    for (const auto& box : tiles) {
        bool merged_flag = false;

        for (auto& m : merged) {
            if (overlap_or_close(m, box, gap)) {
                m = merge_bbox(m, box);
                merged_flag = true;
                break;
            }
        }

        if (!merged_flag) {
            merged.push_back(box);
        }
    }
}