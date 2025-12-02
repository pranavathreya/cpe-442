// =======================
// DisplayVideo.cpp (Optimized)
// =======================
// Real-time multithreaded NEON grayscale + Sobel with thread pool.
// Uses row partitioning and zero dynamic allocations in loop.
// PAPI timing + FPS overlay.
//
// Reference to original user file: :contentReference[oaicite:1]{index=1}
// =======================

#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <arm_neon.h>
#include <papi.h>
#include <atomic>
#include <chrono>

#define NTHREADS 4

using namespace cv;

// =====================================================================
// Thread-pool shared structures
// =====================================================================
struct Task {
    Mat* src;
    Mat* dst;
    int start_row;
    int end_row;
    bool do_gray;
    bool do_sobel;
};

Task tasks[NTHREADS];
pthread_t workers[NTHREADS];
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

bool work_ready = false;
bool stop_all = false;
std::atomic<int> finished_threads(0);

// =====================================================================
// HIGH-PERFORMANCE GRAYSCALE (NEON)
// =====================================================================
inline void grayscale_neon(Mat& src, Mat& dst, int r0, int r1)
{
    for (int r = r0; r < r1; r++)
    {
        uint8_t* in  = src.ptr<uint8_t>(r);
        uint8_t* out = dst.ptr<uint8_t>(r);

        int c = 0;
        int limit = src.cols - 8;

        for (; c <= limit; c += 8)
        {
            uint8x8x3_t pix = vld3_u8(in + 3*c);

            uint16x8_t t = vmull_u8(pix.val[0], vdup_n_u8(29));
            t = vmlal_u8(t, pix.val[1], vdup_n_u8(150));
            t = vmlal_u8(t, pix.val[2], vdup_n_u8(77));

            vst1_u8(out + c, vshrn_n_u16(t, 8));
        }

        for (; c < src.cols; c++)
        {
            Vec3b& p = *(Vec3b*)(in + 3*c);
            out[c] = (p[0] * 29 + p[1] * 150 + p[2] * 77) >> 8;
        }
    }
}

// =====================================================================
// HIGH-PERFORMANCE SOBEL (NEON)
// =====================================================================
inline void sobel_neon(Mat& src, Mat& dst, int r0, int r1)
{
    int rows = src.rows;
    int cols = src.cols;

    r0 = std::max(1, r0);
    r1 = std::min(rows - 1, r1);

    for (int r = r0; r < r1; r++)
    {
        const uint8_t* prev = src.ptr<uint8_t>(r - 1);
        const uint8_t* curr = src.ptr<uint8_t>(r);
        const uint8_t* next = src.ptr<uint8_t>(r + 1);
        uint8_t* out = dst.ptr<uint8_t>(r);

        int c = 1;
        int limit = cols - 9;

        for (; c <= limit; c += 8)
        {
            uint8x8_t pL = vld1_u8(prev + (c - 1));
            uint8x8_t pC = vld1_u8(prev + c);
            uint8x8_t pR = vld1_u8(prev + (c + 1));

            uint8x8_t cL = vld1_u8(curr + (c - 1));
            uint8x8_t cR = vld1_u8(curr + (c + 1));

            uint8x8_t nL = vld1_u8(next + (c - 1));
            uint8x8_t nC = vld1_u8(next + c);
            uint8x8_t nR = vld1_u8(next + (c + 1));

            int16x8_t pL16 = vreinterpretq_s16_u16(vmovl_u8(pL));
            int16x8_t pC16 = vreinterpretq_s16_u16(vmovl_u8(pC));
            int16x8_t pR16 = vreinterpretq_s16_u16(vmovl_u8(pR));

            int16x8_t cL16 = vreinterpretq_s16_u16(vmovl_u8(cL));
            int16x8_t cR16 = vreinterpretq_s16_u16(vmovl_u8(cR));

            int16x8_t nL16 = vreinterpretq_s16_u16(vmovl_u8(nL));
            int16x8_t nC16 = vreinterpretq_s16_u16(vmovl_u8(nC));
            int16x8_t nR16 = vreinterpretq_s16_u16(vmovl_u8(nR));

            int16x8_t gx = vsubq_s16(pR16, pL16);
            gx = vaddq_s16(gx, vshlq_n_s16(vsubq_s16(cR16, cL16), 1));
            gx = vaddq_s16(gx, vsubq_s16(nR16, nL16));

            int16x8_t top = vaddq_s16(pL16, pR16);
            top = vaddq_s16(top, vshlq_n_s16(pC16, 1));
            int16x8_t bot = vaddq_s16(nL16, nR16);
            bot = vaddq_s16(bot, vshlq_n_s16(nC16, 1));

            int16x8_t gy = vsubq_s16(bot, top);

            int16x8_t agx = vabsq_s16(gx);
            int16x8_t agy = vabsq_s16(gy);
            int16x8_t mag = vqaddq_s16(agx, agy);

            uint8x8_t mag8 = vqmovn_u16(vreinterpretq_u16_s16(mag));

            vst1_u8(out + c, mag8);
        }

        for (; c < cols - 1; c++)
        {
            int gx = 
                (curr[c+1] - curr[c-1]) * 2 +
                (prev[c+1] - prev[c-1]) +
                (next[c+1] - next[c-1]);

            int gy =
                (next[c-1] + 2*next[c] + next[c+1]) -
                (prev[c-1] + 2*prev[c] + prev[c+1]);

            int m = std::min(255, abs(gx) + abs(gy));
            out[c] = m;
        }
    }
}

// =====================================================================
// Worker thread (persistent)
// =====================================================================
void* worker(void* arg)
{
    int id = (intptr_t)arg;

    while (true)
    {
        pthread_mutex_lock(&mtx);
        while (!work_ready)
            pthread_cond_wait(&cond, &mtx);

        if (stop_all)
        {
            pthread_mutex_unlock(&mtx);
            return nullptr;
        }

        Task& t = tasks[id];
        pthread_mutex_unlock(&mtx);

        if (t.do_gray)
            grayscale_neon(*t.src, *t.dst, t.start_row, t.end_row);
        if (t.do_sobel)
            sobel_neon(*t.src, *t.dst, t.start_row, t.end_row);

        finished_threads++;
    }
    return nullptr;
}

// =====================================================================
// MAIN
// =====================================================================
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("usage: DisplayVideo <Video_Path>\n");
        return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        printf("Error opening video.\n");
        return -1;
    }
    printf("Video opened.\n");

    // Start thread pool
    for (int i = 0; i < NTHREADS; i++)
        pthread_create(&workers[i], nullptr, worker, (void*)(intptr_t)i);

    Mat frame;
    cap.read(frame);

    int H = frame.rows;
    int W = frame.cols;

    Mat gray(H, W, CV_8UC1);
    Mat sobel(H, W, CV_8UC1);

    long long start_t = PAPI_get_real_usec();
    float fps = 0;
    int counter = 0;

    while (true)
    {
        if (!cap.read(frame))
            break;

        int step = H / NTHREADS;
        finished_threads = 0;

        for (int i = 0; i < NTHREADS; i++)
        {
            tasks[i].src = &frame;
            tasks[i].dst = &gray;
            tasks[i].start_row = i * step;
            tasks[i].end_row = (i == NTHREADS - 1 ? H : (i+1)*step);
            tasks[i].do_gray = true;
            tasks[i].do_sobel = false;
        }

        pthread_mutex_lock(&mtx);
        work_ready = true;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mtx);

        while (finished_threads != NTHREADS) {}

        finished_threads = 0;

        for (int i = 0; i < NTHREADS; i++)
        {
            tasks[i].src = &gray;
            tasks[i].dst = &sobel;
            tasks[i].do_gray = false;
            tasks[i].do_sobel = true;
        }

        pthread_mutex_lock(&mtx);
        work_ready = true;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mtx);

        while (finished_threads != NTHREADS) {}

        counter++;
        if (counter >= 10)
        {
            long long now = PAPI_get_real_usec();
            fps = 10.0 / ((now - start_t) * 1e-6);
            start_t = now;
            counter = 0;
        }

        char buf[64];
        snprintf(buf, 64, "FPS: %.2f", fps);
        putText(sobel, buf, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 2);

        imshow("Sobel", sobel);
        if (waitKey(1) == 'q')
            break;
    }

    // stop threads
    pthread_mutex_lock(&mtx);
    stop_all = true;
    work_ready = true;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mtx);

    for (int i = 0; i < NTHREADS; i++)
        pthread_join(workers[i], nullptr);

    return 0;
}

