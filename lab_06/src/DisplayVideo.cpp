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
    bool run_filter;
    //bool run_sobel;
};

inline void sobelTask(const Task& t);
void* worker(void* arg);
inline uint8x8_t gray_scale(uint8x8x3_t rgb_row);

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
    
    int retval;

    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init error: %s\n", PAPI_strerror(retval));
        return -1;
    }

    // Enable PAPI for multithreaded use (with pthreads)
    retval = PAPI_thread_init((unsigned long (*)(void)) (pthread_self));
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_thread_init error: %s\n", PAPI_strerror(retval));
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

    //Mat gray_frame(height, width, CV_8UC1);
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
            tasks[i].dst       = &sobel_frame;
            tasks[i].start_row = i * step;
            tasks[i].end_row   = (i == NTHREADS - 1) ? height : (i + 1) * step;
            tasks[i].run_filter  = true;
            //tasks[i].run_sobel = false;
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

        //// 2) SOBEL PHASE ----------------------------------------
        //pthread_mutex_lock(&mtx);
        //for (int i = 0; i < NTHREADS; i++) {
        //    tasks[i].src       = &gray_frame;
        //    tasks[i].dst       = &sobel_frame;
        //    // Same row partition
        //    tasks[i].start_row = i * step;
        //    tasks[i].end_row   = (i == NTHREADS - 1) ? height : (i + 1) * step;
        //    tasks[i].run_gray  = false;
        //    tasks[i].run_sobel = true;
        //}
        //pending    = NTHREADS;
        //work_ready = true;
        //pthread_cond_broadcast(&cond_work);
        //pthread_mutex_unlock(&mtx);

        //// Wait for all threads to finish Sobel
        //pthread_mutex_lock(&mtx);
        //while (pending > 0) {
        //    pthread_cond_wait(&cond_done, &mtx);
        //}
        //pthread_mutex_unlock(&mtx);

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
	    printf("Frame Rate: %.2f\n", fps);  
        }

        //// Draw FPS on sobel_frame
        //char text[50];
        //snprintf(text, sizeof(text), "FPS: %.2f", fps);
        //putText(
        //    sobel_frame,
        //    text,
        //    Point(10, 30),
        //    FONT_HERSHEY_SIMPLEX,
        //    0.8,
        //    Scalar(255),
        //    2
        //);

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
inline void sobelTask(const Task& t)
{
    Mat* rgb_frame = t.src;
    Mat* sobel = t.dst;

    const int rows = rgb_frame->rows;
    const int cols = rgb_frame->cols;

    // Clamp boundaries (Sobel needs neighbors)
    int start_row = std::max(1, t.start_row);
    int end_row   = std::min(rows - 1, t.end_row);

    for (int i = start_row; i < end_row; i++) {
        const uint8_t* prevRow = rgb_frame->ptr<uint8_t>(i - 1);
        const uint8_t* currRow = rgb_frame->ptr<uint8_t>(i);
        const uint8_t* nextRow = rgb_frame->ptr<uint8_t>(i + 1);

        uint8_t* outRow = sobel->ptr<uint8_t>(i);

        int j = 1;
        // Handle vectorizable part: j..j+7 must be valid, with neighbors
        // Need (j+7) + 1 <= cols - 1 => j <= cols - 9
        int j_vec_end = (cols >= 10) ? (cols - 9) : 0;

        for (; j <= j_vec_end; j += 8) {
	    // Each pixel: BGR interleaved, so we load from src + 3*c
            // Load neighbors in 8-wide chunks
            uint8x8x3_t pL_rgb = vld3_u8(prevRow + 3*(j - 1));
            uint8x8x3_t pC_rgb = vld3_u8(prevRow + 3*(j    ));
            uint8x8x3_t pR_rgb = vld3_u8(prevRow + 3*(j + 1));

            uint8x8x3_t cL_rgb = vld3_u8(currRow + 3*(j - 1));
            uint8x8x3_t cR_rgb = vld3_u8(currRow + 3*(j + 1));

            uint8x8x3_t nL_rgb = vld3_u8(nextRow + 3*(j - 1));
            uint8x8x3_t nC_rgb = vld3_u8(nextRow + 3*(j    ));
            uint8x8x3_t nR_rgb = vld3_u8(nextRow + 3*(j + 1));

            // Load neighbors in 8-wide chunks
            uint8x8_t pL = gray_scale(pL_rgb);
            uint8x8_t pC = gray_scale(pC_rgb);
            uint8x8_t pR = gray_scale(pR_rgb);

            uint8x8_t cL = gray_scale(cL_rgb);
            uint8x8_t cR = gray_scale(cR_rgb);

            uint8x8_t nL = gray_scale(nL_rgb);
            uint8x8_t nC = gray_scale(nC_rgb);
            uint8x8_t nR = gray_scale(nR_rgb);

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
void* worker(void* arg) {
    int id = (intptr_t)arg;

    // --- PAPI per-thread setup ---
    int retval = PAPI_register_thread();
    if (retval != PAPI_OK) {
        fprintf(stderr, "Thread %d: PAPI_register_thread error: %s\n",
                id, PAPI_strerror(retval));
        // You *can* continue, but counters won't work right.
    }

    int EventSet = PAPI_NULL;
    long long values[2]      = {0, 0};  // current reading
    long long prev_values[2] = {0, 0};  // last reading
    long long accum_L1 = 0;
    long long accum_L2 = 0;
    long long frames_counted = 0;

    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Thread %d: PAPI_create_eventset error: %s\n",
                id, PAPI_strerror(retval));
    }

    // Add L1 & L2 events if supported
    if (PAPI_query_event(PAPI_L1_DCM) == PAPI_OK) {
        retval = PAPI_add_event(EventSet, PAPI_L1_DCM);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Thread %d: PAPI_add_event(L1_DCM) error: %s\n",
                    id, PAPI_strerror(retval));
        }
    } else {
        fprintf(stderr, "Thread %d: PAPI_L1_DCM not supported\n", id);
    }

    if (PAPI_query_event(PAPI_L2_DCM) == PAPI_OK) {
        retval = PAPI_add_event(EventSet, PAPI_L2_DCM);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Thread %d: PAPI_add_event(L2_DCM) error: %s\n",
                    id, PAPI_strerror(retval));
        }
    } else {
        fprintf(stderr, "Thread %d: PAPI_L2_DCM not supported\n", id);
    }

    // Start counters ONCE for the lifetime of this worker
    retval = PAPI_start(EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Thread %d: PAPI_start error: %s\n",
                id, PAPI_strerror(retval));
    }

    // --- normal worker loop ---
    while (true) {
        // Wait for work (your existing condvar/mutex logic)
        pthread_mutex_lock(&mtx);
        while (!work_ready && !stop_all) {
            pthread_cond_wait(&cond_work, &mtx);
        }
        if (stop_all) {
            pthread_mutex_unlock(&mtx);
            break;
        }
        Task t = tasks[id];  // copy task locally
        pthread_mutex_unlock(&mtx);

        // ---- perform the actual work for this frame/phase ----
        // If fused: gray+sobel happens here
        if (t.run_filter) {
            // your processing: grayscale, sobelTask(t), etc.
        }

        // ---- PAPI read: compute per-job delta ----
        retval = PAPI_read(EventSet, values);
        if (retval == PAPI_OK) {
            long long dL1 = values[0] - prev_values[0];
            long long dL2 = values[1] - prev_values[1];
            prev_values[0] = values[0];
            prev_values[1] = values[1];

            // Accumulate totals
            accum_L1 += dL1;
            accum_L2 += dL2;
            frames_counted++;

            // Optional: only print every N frames to reduce overhead
            const int N = 60;
            if (frames_counted % N == 0) {
                printf("Thread %d (last %d frames): L1/frame ≈ %.1f, L2/frame ≈ %.1f\n",
                       id, N,
                       (double)accum_L1 / (double)frames_counted,
                       (double)accum_L2 / (double)frames_counted);
            }
        } else {
            fprintf(stderr, "Thread %d: PAPI_read error: %s\n",
                    id, PAPI_strerror(retval));
        }

        // ---- notify main that this thread finished its part ----
        pthread_mutex_lock(&mtx);
        pending--;
        if (pending == 0) {
            work_ready = false;
            pthread_cond_signal(&cond_done);
        }
        pthread_mutex_unlock(&mtx);
    }

    // At worker exit: stop and clean up
    retval = PAPI_stop(EventSet, values);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Thread %d: PAPI_stop error: %s\n",
                id, PAPI_strerror(retval));
    }

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_unregister_thread();

    printf("Thread %d total: frames=%lld, L1=%lld, L2=%lld\n",
           id, frames_counted, accum_L1, accum_L2);

    return nullptr;
}


inline uint8x8_t gray_scale(uint8x8x3_t rgb_row) {
    // temp = 29*R + 150*G + 77*B (approx luminance)
    uint16x8_t temp = vmull_u8(rgb_row.val[2], vdup_n_u8(29));   // R
    temp            = vmlal_u8(temp, rgb_row.val[1], vdup_n_u8(150)); // G
    temp            = vmlal_u8(temp, rgb_row.val[0], vdup_n_u8(77));  // B

    // >> 8 to normalize
    uint8x8_t gray_row = vshrn_n_u16(temp, 8);

    return gray_row;
}
