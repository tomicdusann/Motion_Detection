#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>

#include "constants.h"
#include "pixel-lvl-image-ops-thread-pipeline.h"
#include "morphology-thread-pipeline.h"
#include "bbox_tile.h"
#include "bbox_merge.h"

using namespace std;

// --------------------------- Globals ---------------------------
queue<cv::Mat> captureQueue;
queue<pair<cv::Mat, cv::Mat>> processedQueue; // frame + mask

mutex capMutex, procMutex;
condition_variable capCV, procCV;

bool finished = false;
const int MAX_QUEUE_SIZE = 5;

// --------------------------- Capture Thread ---------------------------
void captureThreadFunc(cv::VideoCapture& cap) {
    cv::Mat frame;
    while (true) {
        {
            unique_lock<mutex> lock(capMutex);  // capture thread uzima propusnicu (lock) kako bi mogao da zakljuca kondicionu varijablu
            capCV.wait(lock, [] { return captureQueue.size() < MAX_QUEUE_SIZE; });  // kondiciona varijabla stopira izvrsavanje niti i nit privremeno pusta propusnicu sve dok lambda ne bude tacna, nakon toga opet uzima propusnicu i nastavlja sa radom
        }

        if (!cap.read(frame)) break;    // ako nema citanja prekini

        {
            unique_lock<mutex> lock(capMutex);  // nit uzima propusnicu(lock) kako bi upisala frejmu u captureQueue
            captureQueue.push(frame.clone());
        }
        capCV.notify_all(); // obavestavaju se ostale pokrenute niti koje cekaju na capCV da mogu da krenu sa izvrsenjem/provere da li mogu da nastave, budi ih
    }

    finished = true; // kada se iscitaju svi frejmovi iz videa postavi flag finished na true
    capCV.notify_all(); 
    procCV.notify_all();
}

// --------------------------- Processing Worker ---------------------------
void processingWorker(cv::Mat bg) {
    cv::Mat gray, mask, tmp, eroded;
    while (true) {
        cv::Mat frame;
        {
            unique_lock<mutex> lock(capMutex); // worker uzima propusnicu(lock) da moze da zakljuca kondicionu varijablu
            capCV.wait(lock, [] { return !captureQueue.empty() || finished; }); // worker je stopiran sve dok je captureQueue prazan ili dok nije finished 
            if (captureQueue.empty() && finished) break;
            frame = captureQueue.front();   // vadi frejmove iz queue
            captureQueue.pop(); // sklanja ih iz queuea kad ih izvuce
        }
        capCV.notify_all(); // obavesti sve ostale niti koje zavise od ove kondicione varijable da mogu da nastave dalje tj. da provere da li mogu dalje

        if (gray.empty()) {
            gray.create(frame.size(), CV_8UC1);
            mask.create(frame.size(), CV_8UC1);
            tmp.create(frame.size(), CV_8UC1);
            eroded.create(frame.size(), CV_8UC1);
            bg = cv::Mat::zeros(frame.size(), CV_8UC1);
        }

        // ---------------- Processing ----------------
        bgr_to_gray(frame, gray);
        absdiff_thresh(gray, bg, mask, DIFF_THRESHOLD);
        erosion(mask, eroded);
        dilation(eroded, eroded);
        edge_thinning(mask, eroded, mask);
        blur_h(mask, tmp);
        blur_v(tmp, mask);
        update_bg(gray, bg, BG_ALPHA);

        vector<BBox> tiles;
        find_tile_bboxes(mask, tiles);

        vector<BBox> filtered;
        for (const auto& b : tiles) {
            int area = (b.x2 - b.x1) * (b.y2 - b.y1);
            if (area >= MIN_TILE_AREA) filtered.push_back(b);
        }
        tiles = move(filtered);

        vector<BBox> objects;
        merge_tile_bboxes(tiles, objects, MERGE_GAP);

        for (const auto& b : objects) {
            cv::rectangle(frame,
                cv::Point(b.x1, b.y1),
                cv::Point(b.x2, b.y2),
                { 0, 0, 255 }, 2);
        }

        {
            unique_lock<mutex> lock(procMutex); // worker thread uzima propusnicu(lock) da bi dodavao u processed Queue
            processedQueue.push({ frame, mask });
        }
        procCV.notify_all();    // sve niti koje koriste kondicionu varijablu procCV mogu da nastave sa izvrsavanjem
    }
}

// --------------------------- Display Thread ---------------------------
void displayThreadFunc() {
    using clock = std::chrono::high_resolution_clock;

    auto t0 = clock::now();
    int frames = 0;
    double fps = 0.0;

    while (true) {
        pair<cv::Mat, cv::Mat> data;
        {
            unique_lock<mutex> lock(procMutex); // uzima se propusnica da se manipulise sa procCV kondicionom varijablom
            procCV.wait(lock, [] { return !processedQueue.empty() || finished; });  // nit ceka dok je queue prazan ili dok nije finished flag
            if (processedQueue.empty() && finished) break;
            data = processedQueue.front();
            processedQueue.pop();
        }
        procCV.notify_all();

        // ---------------- FPS logic ----------------
        frames++;
        auto t1 = clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (dt >= 1.0) {
            fps = frames / dt;
            frames = 0;
            t0 = t1;
        }

        cv::putText(
            data.first,
            "FPS: " + std::to_string((int)fps),
            { 20, 40 },
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            { 0, 255, 0 },
            2
        );

        cv::imshow("CPU Frame", data.first);
        cv::imshow("CPU Mask", data.second);
        if (cv::waitKey(1) == 27) break;
    }
}

// --------------------------- Main ---------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: cpu_motion <video>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) 
        return -1;

    cv::Mat bg;

    // Capture thread
    thread captureThread(captureThreadFunc, ref(cap));

    // Processing thread pool
    const int numWorkers = 6; // 4 niti radnika
    vector<thread> workers;
    for (int i = 0; i < numWorkers; i++) {
        workers.emplace_back(processingWorker, bg.clone()); // preko emplace_back pravimo thread objekat i u isto vreme ga stavljamo u vector
    }

    // Display thread
    thread displayThread(displayThreadFunc);

    // Join threads
    captureThread.join();

    for (auto& w : workers) 
        w.join();

    displayThread.join();

    return 0;
}
