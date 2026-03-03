#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(err) if((err) != cudaSuccess){ \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
    << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1); }

#define BLOCK 16    // CUDA block za image operacije - njegova dimenzija
#define K 2
#define TILE 32     // tile size for multi-bbox

struct BBox {
    int x1, y1, x2, y2;
};

bool overlap_or_close(const BBox& a, const BBox& b, int gap)
{
    if (a.x2 + gap < b.x1) return false;
    if (b.x2 + gap < a.x1) return false;
    if (a.y2 + gap < b.y1) return false;
    if (b.y2 + gap < a.y1) return false;
    return true;        // Ako se pravougaonici ne razdvajaju ni po x ni po y osi vise od GAPa — oni su onda jedan pored drugog
}

BBox merge_bbox(const BBox& a, const BBox& b)
{
    return {
        std::min(a.x1, b.x1),
        std::min(a.y1, b.y1),
        std::max(a.x2, b.x2),
        std::max(a.y2, b.y2)
    };  // vracamo najmanji moguci bbox koji obuhvata oba objekta kad su sada spojeni
}

// ================= CUDA KERNELS =================

__global__ void bgr_to_gray_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int cols, int rows, int step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int i = y * step + 3 * x;
    dst[y * cols + x] =
        (unsigned char)(0.114f * src[i] + 0.587f * src[i + 1] + 0.299f * src[i + 2]);
}

__global__ void absdiff_thresh_kernel(
    const unsigned char* cur,
    const unsigned char* bg,
    unsigned char* out,
    int cols, int rows,
    unsigned char thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int i = y * cols + x;
    out[i] = (abs(cur[i] - bg[i]) > thresh) ? 255 : 0;      // gledamo da li je razlika u intenzitetu piksela dovoljna da nesto smatramo za foreground(objekat) ili ostaje kao pozadina trenutnog frejma
}

__global__ void temporal_bg_update(     // poenta je da azuriramo background model bg, tako da uvek bude u globalnoj memoriji, spreman za sl. iteraciju
    const unsigned char* cur,
    unsigned char* bg,
    int cols, int rows,
    float alpha)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int i = y * cols + x;
    bg[i] = (unsigned char)(alpha * cur[i] + (1.0f - alpha) * bg[i]);   // formula preko koje prepoznajemo promene u pozadini iz frejma u frejm
}                                                                       // koef. alpha podesiti da se prilagodjava postepeno promenama pozadine a ne skokovito

// -------- OPEN (erosion + dilation) --------          uklanjamo mali sum prvim korakom (min) a vracamo oblik objekta drugim korakom (max)
__global__ void open_fused(
    const unsigned char* src,
    unsigned char* dst,
    int cols, int rows)
{
    __shared__ unsigned char tile[BLOCK + 2 * K][BLOCK + 2 * K];
    __shared__ unsigned char eroded[BLOCK][BLOCK];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * BLOCK + tx;
    int y = blockIdx.y * BLOCK + ty;

    tile[ty + K][tx + K] = (x < cols && y < rows) ? src[y * cols + x] : 255;
    __syncthreads();

    if (x < K || y < K || x >= cols - K || y >= rows - K) return;

    // erosion uklanja sitne sumove, eliminise ivice i ostavlja samo "unutrasnju masu" objekta
    unsigned char mn = 255;
#pragma unroll
    for (int dy = -K; dy <= K; dy++)                        // dx i dy mogu da tumacim kao pomeraj levo-desno (x) i gore-dole (y); dx=-1 znaci idemo levo; dy=+2 znaci idemo 2 na dole
        for (int dx = -K; dx <= K; dx++)                    // od -K do K u smislu 2 u levo, centar i 2 u desno -> 5 piksela; 5x5 je oblast koju gledam     
            mn = min(mn, tile[ty + K + dy][tx + K + dx]);

    eroded[ty][tx] = mn;
    __syncthreads();

    // uklanja sitne sumove ali obrnuto, sad moramo da poboljsamo kvalitet ivice unutrasnjosti objekta
    unsigned char mx = 0;
#pragma unroll
    for (int dy = -K; dy <= K; dy++)
        for (int dx = -K; dx <= K; dx++) {
            int yy = ty + dy, xx = tx + dx;
            if (yy >= 0 && yy < BLOCK && xx >= 0 && xx < BLOCK)
                mx = max(mx, eroded[yy][xx]);
        }

    dst[y * cols + x] = mx;
}

// ovo ce samo da bude za izvlacenje kontura, ako pixel postoji u maski ali ne u eroded-u, to je ivica objekta
__global__ void edge_thinning(
    const unsigned char* mask,
    const unsigned char* eroded,
    unsigned char* out,
    int cols, int rows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int i = y * cols + x;
    out[i] = (mask[i] > eroded[i]) ? 255 : 0;
}

// separable blur jer ako radimo x3 horizontalno pa x3 vertikalno bice brze nego u jednom prolazu, jer bi bilo 9 prolaza
__global__ void blur_h(const unsigned char* src, unsigned char* dst, int c, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= c - 1 || y >= r) return;
    int i = y * c + x;
    dst[i] = (src[i - 1] + src[i] + src[i + 1]) / 3;
}

__global__ void blur_v(const unsigned char* src, unsigned char* dst, int c, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y <= 0 || y >= r - 1 || x >= c) return;
    int i = y * c + x;
    dst[i] = (src[i - c] + src[i] + src[i + c]) / 3;
}

// -------- TILE BASED MULTI BBOX --------
__global__ void tile_bbox_kernel(
    const unsigned char* mask,
    int* tile_bboxes,
    int cols, int rows)
{
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;

    int x0 = tile_x * TILE;
    int y0 = tile_y * TILE;

    // trazim najlevlji, najdesni, najgornji i najdonji piksel koji predstavlja objekat
    int xmin = cols, ymin = rows, xmax = 0, ymax = 0;
    bool found = false;

    // ovo bi trebalo da se izvrsava paralelno ispod haube - nalazimo ekstremne tacke koje ce biti temena nase konture (bbox-a)
    for (int dy = threadIdx.y; dy < TILE; dy += blockDim.y)
        for (int dx = threadIdx.x; dx < TILE; dx += blockDim.x) {
            int x = x0 + dx;
            int y = y0 + dy;
            if (x >= cols || y >= rows) continue;
            if (mask[y * cols + x]) {
                xmin = min(xmin, x);
                ymin = min(ymin, y);
                xmax = max(xmax, x);
                ymax = max(ymax, y);
                found = true;
            }
        }

    __shared__ int sxmin, symin, sxmax, symax, svalid;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sxmin = cols; symin = rows;
        sxmax = 0; symax = 0;
        svalid = 0;
    }
    __syncthreads();

    if (found) {
        atomicMin(&sxmin, xmin);
        atomicMin(&symin, ymin);
        atomicMax(&sxmax, xmax);
        atomicMax(&symax, ymax);
        atomicExch(&svalid, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && svalid) {
        int idx = (tile_y * gridDim.x + tile_x) * 4;
        tile_bboxes[idx + 0] = sxmin;
        tile_bboxes[idx + 1] = symin;
        tile_bboxes[idx + 2] = sxmax;
        tile_bboxes[idx + 3] = symax;
    }
}

// ================= MAIN =================

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: gpu_motion_stable_tile_bbox <video>\n";
        return -1;
    }

    std::vector<BBox> tiles;

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) return -1;

    int rows = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int cols = cap.get(cv::CAP_PROP_FRAME_WIDTH);

    size_t graySize = rows * cols;
    size_t bgrSize = rows * cols * 3;

    unsigned char* d_bgr, * d_gray, * d_bg, * d_mask, * d_tmp, * d_eroded;
    cudaMalloc(&d_bgr, bgrSize);
    cudaMalloc(&d_gray, graySize);
    cudaMalloc(&d_bg, graySize);
    cudaMalloc(&d_mask, graySize);
    cudaMalloc(&d_tmp, graySize);
    cudaMalloc(&d_eroded, graySize);
    cudaMemset(d_bg, 0, graySize);

    int tiles_x = (cols + TILE - 1) / TILE;
    int tiles_y = (rows + TILE - 1) / TILE;
    int num_tiles = tiles_x * tiles_y;

    int* d_tile_bboxes;
    cudaMalloc(&d_tile_bboxes, num_tiles * 4 * sizeof(int));
    cudaMemset(d_tile_bboxes, 0, num_tiles * 4 * sizeof(int));

    dim3 block(BLOCK, BLOCK);
    dim3 grid((cols + BLOCK - 1) / BLOCK, (rows + BLOCK - 1) / BLOCK);
    dim3 tileGrid(tiles_x, tiles_y);
    dim3 tileBlock(8, 8);   // manji blokovi, manje niti - manje zauzete registrske i shared memorije, zahtevnost operacije nam to dozvoljava

    unsigned char threshold = 30;
    float alpha = 0.05f;

    cv::Mat frame, mask(rows, cols, CV_8UC1);

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    int frames = 0;
    double fps = 0;

    while (cap.read(frame)) {

        cudaMemcpy(d_bgr, frame.data, bgrSize, cudaMemcpyHostToDevice);

        bgr_to_gray_kernel << <grid, block >> > (d_bgr, d_gray, cols, rows, cols * 3);
        absdiff_thresh_kernel << <grid, block >> > (d_gray, d_bg, d_mask, cols, rows, threshold);
        open_fused << <grid, block >> > (d_mask, d_eroded, cols, rows);
        edge_thinning << <grid, block >> > (d_mask, d_eroded, d_mask, cols, rows);
        blur_h << <grid, block >> > (d_mask, d_tmp, cols, rows);
        blur_v << <grid, block >> > (d_tmp, d_mask, cols, rows);
        temporal_bg_update << <grid, block >> > (d_gray, d_bg, cols, rows, alpha);

        cudaMemset(d_tile_bboxes, 0, num_tiles * 4 * sizeof(int));
        tile_bbox_kernel << <tileGrid, tileBlock >> > (d_mask, d_tile_bboxes, cols, rows);

        cudaMemcpy(mask.data, d_mask, graySize, cudaMemcpyDeviceToHost);

        std::vector<int> h_tiles(num_tiles * 4);    // za svaki tile imamo po bbox
        cudaMemcpy(h_tiles.data(), d_tile_bboxes,
            num_tiles * 4 * sizeof(int),
            cudaMemcpyDeviceToHost);

        tiles.clear();

        // --- FILTER TILE BOXES ---
        const int MIN_TILE_AREA = 200;  // minimalna broj piksela koje pokriva bbox da bi se uzeo u razmatranje, DA LI JE SUM?

        for (int i = 0; i < num_tiles; i++) {
            int x1 = h_tiles[i * 4 + 0];
            int y1 = h_tiles[i * 4 + 1];
            int x2 = h_tiles[i * 4 + 2];
            int y2 = h_tiles[i * 4 + 3];

            int area = (x2 - x1) * (y2 - y1);   // a*b
            if (x2 > x1 && y2 > y1 && area > MIN_TILE_AREA) { // ako su uslovi ipsunjeni stavljaj podatke u kontejnter, kandidati za pravi objekat
                tiles.push_back({ x1, y1, x2, y2 });
            }
        }

        // --- MERGE TILES (8-neighborhood) ako objekat zauzima vise TILE-ova---
        std::vector<BBox> merged;   // konacni bbox-ovi
        const int GAP = TILE / 2;

        for (auto& box : tiles) {   // poredim odnos preklapanja i rastojanja kandidata za objekat sa ostalim odlucenim objektima
            bool merged_flag = false;

            for (auto& m : merged) {
                if (overlap_or_close(m, box, GAP)) {
                    m = merge_bbox(m, box);
                    merged_flag = true;
                    break;
                }
            }

            if (!merged_flag)
                merged.push_back(box);  // finalni objekti, ako je uslov FALSE znaci da je objekat samostalan, nije se spajao sa ostalim
        }

        // --- DRAW FINAL BOUNDING BOXES ---
        for (auto& b : merged) {
            cv::rectangle(frame,
                { b.x1, b.y1 },
                { b.x2, b.y2 },
                { 0, 0, 255 }, 2);
        }

        frames++;
        auto t1 = clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (dt >= 1.0) {
            fps = frames / dt;
            frames = 0;
            t0 = t1;
        }

        cv::putText(frame, "FPS: " + std::to_string((int)fps),
            { 20,40 }, cv::FONT_HERSHEY_SIMPLEX, 1, { 0,255,0 }, 2);

        cv::imshow("GPU Frame", frame);
        cv::imshow("Stable Mask", mask);
        if (cv::waitKey(1) == 27) break;
    }

    cudaFree(d_bgr);
    cudaFree(d_gray);
    cudaFree(d_bg);
    cudaFree(d_mask);
    cudaFree(d_tmp);
    cudaFree(d_eroded);
    cudaFree(d_tile_bboxes);

    return 0;
}