#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <arm_neon.h>
#include <papi.h>
#include <algorithm>

#define NTHREADS 4

using namespace cv;

struct Task {
    Mat* src;
    Mat* dst;
    int start_row;
    int end_row;
    bool run_gray;
    bool run_sobel;
};

void sobelTask(const Task& t);
void* worker(void* arg);

// Globals for worker coordination
Task tasks[NTHREADS];
pthread_t workers[NTHREADS];

pthread_mutex_t mtx        = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cond_work  = PTHREAD_COND_INITIALIZER;
pthread_cond_t  cond_done  = PTHREAD_COND_INITIALIZER;

bool work_ready = false;   // Workers should start processing
bool stop_all   = false;   // Signal workers to exit
int  pending    = 0;       // Number of threads still working on current job

// -----------------------------------------------------------
// MAIN
// -----------------------------------------------------------
int main(int argc, char** argv)
{
    int   count      = 0;
    long long start_time = 0;
    long long stop_time  = 0;
    float fps       = 0.0f;

    if (argc != 2) {
        printf("usage: DisplayVideo <Video_Path>\n");
        return -1;
    }

    // Init PAPI (for timing)
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init error!\n");
        return -1;
    }

    // Open the video file
    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        printf("Error: Could not open video file.\n");
        return -1;
    } else {
        printf("Video file opened successfully\n");
    }

    Mat frame;

    // Read first frame to determine dimensions
    if (!cap.read(frame) || frame.empty()) {
        printf("Error: Could not read first frame.\n");
        return -1;
    }

    int width  = frame.cols;
    int height = frame.rows;
    printf("Video resolution: %dx%d\n", width, height);

    if (height < NTHREADS) {
        printf("Error: video height (%d) < NTHREADS (%d).\n", height, NTHREADS);
        return -1;
    }

    Mat gray_frame(height, width, CV_8UC1);
    Mat sobel_frame(height, width, CV_8UC1);

    int step = height / NTHREADS;

    // Create worker threads
    for (int i = 0; i < NTHREADS; ++i) {
        pthread_create(&workers[i], NULL, worker, (void*)(intptr_t)i);
    }

    // We already consumed the first frame; process it then continue
    cap.set(CAP_PROP_POS_FRAMES, 0); // rewind to start
    count = 0;
    start_time = PAPI_get_real_usec();

    while (true) {
        bool ret = cap.read(frame);
        if (!ret || frame.empty()) {
            printf("Last frame reached or error.\n");
            break;
        }

        // 1) GRAYSCALE PHASE ------------------------------------
        pthread_mutex_lock(&mtx);
        for (int i = 0; i < NTHREADS; i++) {
            tasks[i].src       = &frame;
            tasks[i].dst       = &gray_frame;
            tasks[i].start_row = i * step;
            tasks[i].end_row   = (i == NTHREADS - 1) ? height : (i + 1) * step;
            tasks[i].run_gray  = true;
            tasks[i].run_sobel = false;
        }
        pending    = NTHREADS;
        work_ready = true;
        pthread_cond_broadcast(&cond_work);
        pthread_mutex_unlock(&mtx);

        // Wait for all threads to finish grayscale
        pthread_mutex_lock(&mtx);
        while (pending > 0) {
            pthread_cond_wait(&cond_done, &mtx);
        }
        pthread_mutex_unlock(&mtx);

        // 2) SOBEL PHASE ----------------------------------------
        pthread_mutex_lock(&mtx);
        for (int i = 0; i < NTHREADS; i++) {
            tasks[i].src       = &gray_frame;
            tasks[i].dst       = &sobel_frame;
            // Same row partition
            tasks[i].start_row = i * step;
            tasks[i].end_row   = (i == NTHREADS - 1) ? height : (i + 1) * step;
            tasks[i].run_gray  = false;
            tasks[i].run_sobel = true;
        }
        pending    = NTHREADS;
        work_ready = true;
        pthread_cond_broadcast(&cond_work);
        pthread_mutex_unlock(&mtx);

        // Wait for all threads to finish Sobel
        pthread_mutex_lock(&mtx);
        while (pending > 0) {
            pthread_cond_wait(&cond_done, &mtx);
        }
        pthread_mutex_unlock(&mtx);

        // FPS measurement every 10 frames
        count++;
        if (count >= 10) {
            stop_time = PAPI_get_real_usec();
            long long delta = stop_time - start_time;
            if (delta > 0) {
                // 10 frames per delta microseconds
                fps = 10.0f * 1e6f / (float)delta;
            }
            start_time = stop_time;
            count = 0;
        }

        // Draw FPS on sobel_frame
        char text[50];
        snprintf(text, sizeof(text), "FPS: %.2f", fps);
        putText(
            sobel_frame,
            text,
            Point(10, 30),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(255),
            2
        );

        imshow("Sobel Frame", sobel_frame);

        char c = (char)waitKey(25);
        if (c == 'q' || c == 27) { // 'q' or ESC
            break;
        }
    }

    // Signal threads to stop and join them
    pthread_mutex_lock(&mtx);
    stop_all   = true;
    work_ready = true;  // wake any waiting threads
    pthread_cond_broadcast(&cond_work);
    pthread_mutex_unlock(&mtx);

    for (int i = 0; i < NTHREADS; ++i) {
        pthread_join(workers[i], NULL);
    }

    cap.release();
    destroyAllWindows();
    PAPI_shutdown();
    return 0;
}

// -----------------------------------------------------------
// SOBEL TASK (per-thread row range)
// -----------------------------------------------------------
void sobelTask(const Task& t)
{
    Mat* gray  = t.src;
    Mat* sobel = t.dst;

    const int rows = gray->rows;
    const int cols = gray->cols;

    // Clamp boundaries (Sobel needs neighbors)
    int start_row = std::max(1, t.start_row);
    int end_row   = std::min(rows - 1, t.end_row);

    for (int i = start_row; i < end_row; i++) {
        const uint8_t* prevRow = gray->ptr<uint8_t>(i - 1);
        const uint8_t* currRow = gray->ptr<uint8_t>(i);
        const uint8_t* nextRow = gray->ptr<uint8_t>(i + 1);

        uint8_t* outRow = sobel->ptr<uint8_t>(i);

        int j = 1;
        // Handle vectorizable part: j..j+7 must be valid, with neighbors
        // Need (j+7) + 1 <= cols - 1 => j <= cols - 9
        int j_vec_end = (cols >= 10) ? (cols - 9) : 0;

        for (; j <= j_vec_end; j += 8) {
            // Load neighbors in 8-wide chunks
            uint8x8_t pL = vld1_u8(prevRow + (j - 1));
            uint8x8_t pC = vld1_u8(prevRow + (j    ));
            uint8x8_t pR = vld1_u8(prevRow + (j + 1));

            uint8x8_t cL = vld1_u8(currRow + (j - 1));
            uint8x8_t cR = vld1_u8(currRow + (j + 1));

            uint8x8_t nL = vld1_u8(nextRow + (j - 1));
            uint8x8_t nC = vld1_u8(nextRow + (j    ));
            uint8x8_t nR = vld1_u8(nextRow + (j + 1));

            // Extend to signed 16-bit
            int16x8_t pL16 = vreinterpretq_s16_u16(vmovl_u8(pL));
            int16x8_t pC16 = vreinterpretq_s16_u16(vmovl_u8(pC));
            int16x8_t pR16 = vreinterpretq_s16_u16(vmovl_u8(pR));

            int16x8_t cL16 = vreinterpretq_s16_u16(vmovl_u8(cL));
            int16x8_t cR16 = vreinterpretq_s16_u16(vmovl_u8(cR));

            int16x8_t nL16 = vreinterpretq_s16_u16(vmovl_u8(nL));
            int16x8_t nC16 = vreinterpretq_s16_u16(vmovl_u8(nC));
            int16x8_t nR16 = vreinterpretq_s16_u16(vmovl_u8(nR));

            // gx = (pR - pL) + 2*(cR - cL) + (nR - nL)
            int16x8_t gx = vsubq_s16(pR16, pL16);
            int16x8_t cdiff = vsubq_s16(cR16, cL16);
            gx = vaddq_s16(gx, vshlq_n_s16(cdiff, 1)); // *2
            gx = vaddq_s16(gx, vsubq_s16(nR16, nL16));

            // gy = (nL + 2*nC + nR) − (pL + 2*pC + pR)
            int16x8_t top = vaddq_s16(pL16, pR16);
            top = vaddq_s16(top, vshlq_n_s16(pC16, 1));

            int16x8_t bot = vaddq_s16(nL16, nR16);
            bot = vaddq_s16(bot, vshlq_n_s16(nC16, 1));

            int16x8_t gy = vsubq_s16(bot, top);

            // |gx| + |gy| with saturation to 8-bit
            int16x8_t mag16 = vqaddq_s16(vabsq_s16(gx), vabsq_s16(gy));
            uint8x8_t mag8  = vqmovn_u16(vreinterpretq_u16_s16(mag16));

            vst1_u8(outRow + j, mag8);
        }

        // Scalar tail for remaining pixels
        for (; j < cols - 1; j++) {
            int gx = (currRow[j+1] - currRow[j-1]) * 2
                   + (prevRow[j+1] - prevRow[j-1])
                   + (nextRow[j+1] - nextRow[j-1]);

            int gy = (nextRow[j-1] + 2*nextRow[j] + nextRow[j+1])
                   - (prevRow[j-1] + 2*prevRow[j] + prevRow[j+1]);

            int mag = std::min(255, std::abs(gx) + std::abs(gy));
            outRow[j] = (uint8_t)mag;
        }
    }
}

// -----------------------------------------------------------
// WORKER THREAD
// -----------------------------------------------------------
void* worker(void* arg)
{
    int id = (intptr_t)arg;

    while (true) {
        // Wait for work or stop signal
        pthread_mutex_lock(&mtx);
        while (!work_ready && !stop_all) {
            pthread_cond_wait(&cond_work, &mtx);
        }

        if (stop_all) {
            pthread_mutex_unlock(&mtx);
            break;
        }

        // Make a local copy of the task to avoid races
        Task t = tasks[id];
        // Mark that this thread has consumed current "work_ready" flag.
        // All threads share the same flag, but we will turn it off in main
        // only after all have finished (pending == 0).
        pthread_mutex_unlock(&mtx);

        // Run grayscale if requested
        if (t.run_gray && t.src && t.dst) {
            for (int r = t.start_row; r < t.end_row; ++r) {
                const uint8_t* src = t.src->ptr<uint8_t>(r);
                uint8_t* dst       = t.dst->ptr<uint8_t>(r);

                int cols = t.src->cols;
                int c = 0;

                // Vectorized: 8 pixels at a time
                for (; c <= cols - 8; c += 8) {
                    // Each pixel: BGR interleaved, so we load from src + 3*c
                    uint8x8x3_t pix = vld3_u8(src + 3 * c);

                    // temp = 29*R + 150*G + 77*B (approx luminance)
                    uint16x8_t temp = vmull_u8(pix.val[2], vdup_n_u8(29));   // R
                    temp            = vmlal_u8(temp, pix.val[1], vdup_n_u8(150)); // G
                    temp            = vmlal_u8(temp, pix.val[0], vdup_n_u8(77));  // B

                    // >> 8 to normalize
                    uint8x8_t gray = vshrn_n_u16(temp, 8);
                    vst1_u8(dst + c, gray);
                }

                // Scalar tail
                for (; c < cols; ++c) {
                    uint8_t b = src[3 * c + 0];
                    uint8_t g = src[3 * c + 1];
                    uint8_t r = src[3 * c + 2];
                    dst[c] = (uint8_t)((29 * r + 150 * g + 77 * b) >> 8);
                }
            }
        }

        // Run Sobel if requested
        if (t.run_sobel && t.src && t.dst) {
            sobelTask(t);
        }

        // Notify main that this thread finished its portion
        pthread_mutex_lock(&mtx);
        pending--;
        if (pending == 0) {
            // All threads finished current job
            work_ready = false;
            pthread_cond_signal(&cond_done);
        }
        pthread_mutex_unlock(&mtx);
    }

    return nullptr;
}

