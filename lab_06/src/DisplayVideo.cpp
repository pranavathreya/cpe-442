#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;

static const int NUM_THREADS = 4;

// ----------------------------------------------------------
// Job description for each worker
// ----------------------------------------------------------
enum class JobType {
    None,
    Grayscale,
    Sobel,
    Exit
};

struct Job {
    Mat* input;
    Mat* output;
    int rowStart;
    int rowEnd;
    JobType type;
};

// ----------------------------------------------------------
// Shared state for thread pool
// ----------------------------------------------------------
static Job       g_jobs[NUM_THREADS];
static pthread_t g_threads[NUM_THREADS];

static pthread_mutex_t g_jobMutex  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_jobCond   = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_doneCond  = PTHREAD_COND_INITIALIZER;

static std::atomic<int> g_jobsRemaining(0);
static bool g_newWorkAvailable = false;
static bool g_shutdownRequested = false;

// ----------------------------------------------------------
// NEON Grayscale: BGR (8UC3) -> Gray (8UC1)
// Each thread handles rows [rowStart, rowEnd)
// ----------------------------------------------------------
static void grayscale_neon(Mat& src, Mat& dst, int rowStart, int rowEnd)
{
    const int width = src.cols;

    for (int y = rowStart; y < rowEnd; ++y) {
        const uint8_t* srcRow = src.ptr<uint8_t>(y);
        uint8_t*       dstRow = dst.ptr<uint8_t>(y);

        int x = 0;
        // Process 8 pixels at a time
        for (; x <= width - 8; x += 8) {
            // Load 8 pixels (interleaved BGR)
            uint8x8x3_t bgr = vld3_u8(srcRow + 3 * x);

            // Convert to wider type
            uint16x8_t b16 = vmovl_u8(bgr.val[0]);
            uint16x8_t g16 = vmovl_u8(bgr.val[1]);
            uint16x8_t r16 = vmovl_u8(bgr.val[2]);

            // Weighted sum: Y ≈ 0.114 B + 0.587 G + 0.299 R
            // Using integer weights scaled by 256: 29, 150, 77
            uint16x8_t y16 = vmulq_n_u16(b16, 29);
            y16 = vmlaq_n_u16(y16, g16, 150);
            y16 = vmlaq_n_u16(y16, r16, 77);

            // Downscale back to 8-bit
            uint8x8_t y8 = vshrn_n_u16(y16, 8);
            vst1_u8(dstRow + x, y8);
        }

        // Scalar tail for leftover pixels
        for (; x < width; ++x) {
            const uint8_t* p = srcRow + 3 * x;
            int b = p[0];
            int g = p[1];
            int r = p[2];
            int y = (29 * b + 150 * g + 77 * r) >> 8;
            dstRow[x] = static_cast<uint8_t>(y);
        }
    }
}

// ----------------------------------------------------------
// NEON Sobel on grayscale 8UC1 -> 8UC1
// Each thread handles rows [rowStart, rowEnd)
// Border rows are handled by clamping to [1, rows-2]
// ----------------------------------------------------------
static void sobel_neon(Mat& src, Mat& dst, int rowStart, int rowEnd)
{
    const int rows = src.rows;
    const int cols = src.cols;

    // Avoid top/bottom border (need y-1 and y+1)
    int y0 = max(rowStart, 1);
    int y1 = min(rowEnd, rows - 1);

    for (int y = y0; y < y1; ++y) {
        const uint8_t* rowAbove = src.ptr<uint8_t>(y - 1);
        const uint8_t* rowCurr  = src.ptr<uint8_t>(y);
        const uint8_t* rowBelow = src.ptr<uint8_t>(y + 1);
        uint8_t* outRow         = dst.ptr<uint8_t>(y);

        int x = 1;                     // avoid left border
        int vecLimit = cols - 8 - 1;   // keep margin for x+7, x+1, x-1

        // Vectorized part
        for (; x <= vecLimit; x += 8) {
            // Load left/center/right neighborhoods above, center, below
            uint8x8_t aL = vld1_u8(rowAbove + x - 1);
            uint8x8_t aC = vld1_u8(rowAbove + x);
            uint8x8_t aR = vld1_u8(rowAbove + x + 1);

            uint8x8_t cL = vld1_u8(rowCurr + x - 1);
            uint8x8_t cC = vld1_u8(rowCurr + x);
            uint8x8_t cR = vld1_u8(rowCurr + x + 1);

            uint8x8_t bL = vld1_u8(rowBelow + x - 1);
            uint8x8_t bC = vld1_u8(rowBelow + x);
            uint8x8_t bR = vld1_u8(rowBelow + x + 1);

            // Widen to signed 16-bit
            int16x8_t aL16 = vreinterpretq_s16_u16(vmovl_u8(aL));
            int16x8_t aC16 = vreinterpretq_s16_u16(vmovl_u8(aC));
            int16x8_t aR16 = vreinterpretq_s16_u16(vmovl_u8(aR));

            int16x8_t cL16 = vreinterpretq_s16_u16(vmovl_u8(cL));
            int16x8_t cC16 = vreinterpretq_s16_u16(vmovl_u8(cC));
            int16x8_t cR16 = vreinterpretq_s16_u16(vmovl_u8(cR));

            int16x8_t bL16 = vreinterpretq_s16_u16(vmovl_u8(bL));
            int16x8_t bC16 = vreinterpretq_s16_u16(vmovl_u8(bC));
            int16x8_t bR16 = vreinterpretq_s16_u16(vmovl_u8(bR));

            // Sobel Gx:
            //   gx = (aR + 2*cR + bR) - (aL + 2*cL + bL)
            int16x8_t leftCol  = vaddq_s16(aL16, bL16);
            leftCol            = vaddq_s16(leftCol, vshlq_n_s16(cL16, 1));
            int16x8_t rightCol = vaddq_s16(aR16, bR16);
            rightCol           = vaddq_s16(rightCol, vshlq_n_s16(cR16, 1));
            int16x8_t gx       = vsubq_s16(rightCol, leftCol);

            // Sobel Gy:
            //   gy = (bL + 2*bC + bR) - (aL + 2*aC + aR)
            int16x8_t topRow    = vaddq_s16(aL16, aR16);
            topRow              = vaddq_s16(topRow, vshlq_n_s16(aC16, 1));
            int16x8_t bottomRow = vaddq_s16(bL16, bR16);
            bottomRow           = vaddq_s16(bottomRow, vshlq_n_s16(bC16, 1));
            int16x8_t gy        = vsubq_s16(bottomRow, topRow);

            // |gx| + |gy| with saturation
            int16x8_t agx = vabsq_s16(gx);
            int16x8_t agy = vabsq_s16(gy);
            int16x8_t mag = vqaddq_s16(agx, agy);

            uint8x8_t mag8 = vqmovn_u16(vreinterpretq_u16_s16(mag));
            vst1_u8(outRow + x, mag8);
        }

        // Scalar tail for remaining columns (up to cols-2)
        for (; x < cols - 1; ++x) {
            int aL = rowAbove[x - 1];
            int aC = rowAbove[x];
            int aR = rowAbove[x + 1];

            int cL = rowCurr[x - 1];
            int cC = rowCurr[x];
            int cR = rowCurr[x + 1];

            int bL = rowBelow[x - 1];
            int bC = rowBelow[x];
            int bR = rowBelow[x + 1];

            int gx = (aR + 2 * cR + bR) - (aL + 2 * cL + bL);
            int gy = (bL + 2 * bC + bR) - (aL + 2 * aC + aR);

            int mag = std::abs(gx) + std::abs(gy);
            if (mag > 255) mag = 255;
            outRow[x] = static_cast<uint8_t>(mag);
        }

        // We leave borders (x=0 and x=cols-1) untouched or zero
        outRow[0] = 0;
        outRow[cols - 1] = 0;
    }
}

// ----------------------------------------------------------
// Worker thread function
// ----------------------------------------------------------
static void* worker_main(void* arg)
{
    int id = reinterpret_cast<intptr_t>(arg);

    while (true) {
        pthread_mutex_lock(&g_jobMutex);
        while (!g_newWorkAvailable && !g_shutdownRequested) {
            pthread_cond_wait(&g_jobCond, &g_jobMutex);
        }

        if (g_shutdownRequested) {
            pthread_mutex_unlock(&g_jobMutex);
            break;
        }

        Job job = g_jobs[id];  // copy task locally
        pthread_mutex_unlock(&g_jobMutex);

        if (job.type == JobType::Grayscale) {
            grayscale_neon(*job.input, *job.output, job.rowStart, job.rowEnd);
        } else if (job.type == JobType::Sobel) {
            sobel_neon(*job.input, *job.output, job.rowStart, job.rowEnd);
        } else if (job.type == JobType::Exit) {
            break;
        }

        // Notify completion
        int remaining = --g_jobsRemaining;
        if (remaining == 0) {
            pthread_mutex_lock(&g_jobMutex);
            g_newWorkAvailable = false;
            pthread_cond_signal(&g_doneCond);
            pthread_mutex_unlock(&g_jobMutex);
        }
    }

    return nullptr;
}

// ----------------------------------------------------------
// Helper to dispatch a phase (grayscale or sobel) to all threads
// ----------------------------------------------------------
static void run_phase(JobType type, Mat& src, Mat& dst)
{
    const int rows = src.rows;
    int chunk = rows / NUM_THREADS;
    int rowStart = 0;

    pthread_mutex_lock(&g_jobMutex);
    g_jobsRemaining = NUM_THREADS;
    g_newWorkAvailable = true;

    for (int i = 0; i < NUM_THREADS; ++i) {
        int rowEnd = (i == NUM_THREADS - 1) ? rows : rowStart + chunk;
        g_jobs[i].input    = &src;
        g_jobs[i].output   = &dst;
        g_jobs[i].rowStart = rowStart;
        g_jobs[i].rowEnd   = rowEnd;
        g_jobs[i].type     = type;
        rowStart = rowEnd;
    }

    pthread_cond_broadcast(&g_jobCond);

    // Wait until all threads finish this phase
    while (g_jobsRemaining > 0) {
        pthread_cond_wait(&g_doneCond, &g_jobMutex);
    }
    pthread_mutex_unlock(&g_jobMutex);
}

// ----------------------------------------------------------
// Main program
// ----------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>\n";
        return 1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << argv[1] << "\n";
        return 1;
    }

    // Prime first frame to know dimensions
    Mat firstFrame;
    if (!cap.read(firstFrame) || firstFrame.empty()) {
        std::cerr << "Failed to read first frame.\n";
        return 1;
    }

    int width  = firstFrame.cols;
    int height = firstFrame.rows;

    // Allocate buffers once
    Mat frame   = firstFrame.clone();
    Mat grayImg(height, width, CV_8UC1);
    Mat sobelImg(height, width, CV_8UC1, Scalar(0));

    // Start worker threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_create(&g_threads[i], nullptr, worker_main,
                       reinterpret_cast<void*>(static_cast<intptr_t>(i)));
    }

    using clock_t = chrono::steady_clock;
    auto t0 = clock_t::now();
    int frameCount = 0;
    double fps = 0.0;

    while (true) {
        // We already have first frame; for next iterations read new ones
        if (frameCount > 0) {
            if (!cap.read(frame) || frame.empty()) {
                std::cout << "End of video or read error.\n";
                break;
            }
        }

        // Phase 1: Grayscale
        run_phase(JobType::Grayscale, frame, grayImg);

        // Phase 2: Sobel
        sobelImg.setTo(0);
        run_phase(JobType::Sobel, grayImg, sobelImg);

        // FPS measurement
        frameCount++;
        if (frameCount >= 30) {
            auto t1 = clock_t::now();
            double elapsedSec =
                chrono::duration_cast<chrono::microseconds>(t1 - t0).count()
                / 1e6;
            fps = frameCount / elapsedSec;
            t0 = t1;
            frameCount = 0;
        }

        // Draw FPS on the Sobel image
        char text[64];
        snprintf(text, sizeof(text), "FPS: %.2f", fps);
        putText(sobelImg, text, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 2);

        imshow("Sobel (NEON + pthreads)", sobelImg);
        char key = static_cast<char>(waitKey(1));
        if (key == 'q' || key == 27) { // 'q' or ESC
            break;
        }
    }

    // Request shutdown of workers
    pthread_mutex_lock(&g_jobMutex);
    g_shutdownRequested = true;
    g_newWorkAvailable  = true;
    pthread_cond_broadcast(&g_jobCond);
    pthread_mutex_unlock(&g_jobMutex);

    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(g_threads[i], nullptr);
    }

    return 0;
}

