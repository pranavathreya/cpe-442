#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <arm_neon.h>
#include <papi.h>

#define NTHREADS 4

using namespace cv;
using namespace std::chrono;

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

Task tasks[NTHREADS];
pthread_t workers[NTHREADS];
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool work_ready = false;
bool stop_all = false;

int main(int argc, char** argv)
{
	int count=0;
	long long start_time=0;
	long long stop_time=0;
	float fps=0;
	int height=1080;
	int width=1920;

	bool ret;

    if (argc != 2)
    {
        printf("usage: DisplayVideo <Video_Path>\n");
        return -1;
    }

    // Open the video file
    VideoCapture cap(argv[1]);

    // Check if the video was opened successfully
    if (!cap.isOpened()) {
        printf("Error: Could not open video file.\n");
        return -1;
    } else {
        printf("Video file opened successfully\n");
    }

    Mat frame;

	// Create threads
	for (int i = 0; i < NTHREADS; i++)
			pthread_create(&workers[i], NULL, worker, (void*)(intptr_t)i);

		
	int step = height / NTHREADS;
	Mat gray_frame(height, width, CV_8UC1);
    Mat sobel_frame(height, width, CV_8UC1);
    while (true) {
	if(count==0){
			start_time= PAPI_get_real_usec();
	}	
        ret = cap.read(frame);
        if (!ret) {
            printf("Last frame reached or error.\n");
            break;
        }

		for (int i = 0; i < NTHREADS; i++) {
			tasks[i].src = &frame;
			tasks[i].dst = &gray_frame;
			tasks[i].start_row = i * step;
			tasks[i].end_row = (i == NTHREADS-1) ? height : (i+1)*step;
			tasks[i].run_gray = true;
			tasks[i].run_sobel = false;
		}

		pthread_mutex_lock(&mtx);
		work_ready = true;
		pthread_cond_broadcast(&cond);
		pthread_mutex_unlock(&mtx);
			
		
		/*
        matFrames mfs[NTHREADS];
        for (int q = 0; q < NTHREADS; ++q) {
            mfs[q].src = &frame;
            mfs[q].dst = &gray_frame;
            mfs[q].quarter = q + 1;
            pthread_create(&thread_id[q], NULL, grayScale, (void*)&mfs[q]);
        }
        for (int q = 0; q < NTHREADS; ++q) {
            pthread_join(thread_id[q], NULL);
        }

        matFrames sargs[NTHREADS];
        for (int q = 0; q < NTHREADS; ++q) {
            sargs[q].src = &gray_frame;   // source is the grayscale image
            sargs[q].dst = &sobel_frame;  // destination is the sobel output
            sargs[q].quarter = q + 1;
            pthread_create(&thread_id[q], NULL, sobelThread, (void*)&sargs[q]);
        }
        for (int q = 0; q < NTHREADS; ++q) {
            pthread_join(thread_id[q], NULL);
		}
		*/
		for (int i=0; i<NTHREADS; i++) {
			tasks[i].src = &gray_frame;
			tasks[i].dst = &sobel_frame;
			tasks[i].run_gray = false;
			tasks[i].run_sobel = true;
		}

	    count++;
        if (count >= 10) {
        stop_time = PAPI_get_real_usec();
        fps = 10.0 / ((stop_time - start_time) * pow(10, -6));
        start_time = stop_time;   // reset for next batch
        count = 0;
    }

    // ---- DRAW FPS ON FRAME ----
    char text[50];
    snprintf(text, sizeof(text), "FPS: %.2f", fps);

    cv::putText(
        sobel_frame,              // image
        text,                     // text
        cv::Point(10, 30),        // bottom-left corner
        cv::FONT_HERSHEY_SIMPLEX, // font
        0.8,                      // scale
        cv::Scalar(255),          // color (white for grayscale)
        2                         // thickness
    );

    imshow("Sobel Frame", sobel_frame);

    char c = (char)waitKey(25);
    if (c == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

void sobelTask(const Task& t)
{
    Mat* gray  = t.src;
    Mat* sobel = t.dst;

    const int rows = gray->rows;
    const int cols = gray->cols;

    // Clamp boundaries — avoid first/last row (Sobel needs neighbors)
    int start_row = std::max(1, t.start_row);
    int end_row   = std::min(rows - 1, t.end_row);

    for (int i = start_row; i < end_row; i++)
    {
        const uint8_t* prevRow = gray->ptr<uint8_t>(i - 1);
        const uint8_t* currRow = gray->ptr<uint8_t>(i);
        const uint8_t* nextRow = gray->ptr<uint8_t>(i + 1);

        uint8_t* outRow = sobel->ptr<uint8_t>(i);

        // Vectorizable inner loop:
        // j runs from 1..cols-2; the last valid NEON start is (cols - 9)
        int j = 1;
        const int j_vec_end = (cols >= 9) ? (cols - 9) : 1;

        for (; j <= j_vec_end; j += 8)
        {
            // Load 8-byte chunks from each necessary neighbor
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

            //
            //  gx = (pR - pL) + 2*(cR - cL) + (nR - nL)
            //
            int16x8_t gx = vsubq_s16(pR16, pL16);
            int16x8_t cdiff = vsubq_s16(cR16, cL16);
            gx = vaddq_s16(gx, vshlq_n_s16(cdiff, 1));  // *2
            gx = vaddq_s16(gx, vsubq_s16(nR16, nL16));

            //
            //  gy = (nL + 2*nC + nR) − (pL + 2*pC + pR)
            //
            int16x8_t top = vaddq_s16(pL16, pR16);
            top = vaddq_s16(top, vshlq_n_s16(pC16, 1));

            int16x8_t bot = vaddq_s16(nL16, nR16);
            bot = vaddq_s16(bot, vshlq_n_s16(nC16, 1));

            int16x8_t gy = vsubq_s16(bot, top);

            // |gx| + |gy| → saturated 8-bit
            int16x8_t mag16 = vqaddq_s16(vabsq_s16(gx), vabsq_s16(gy));
            uint8x8_t mag8  = vqmovn_u16(vreinterpretq_u16_s16(mag16));

            vst1_u8(outRow + j, mag8);
        }

        // Handle scalar tail pixels (leftovers)
        for (; j < cols - 1; j++)
        {
            int gx =  (currRow[j+1] - currRow[j-1]) * 2
                    + (prevRow[j+1] - prevRow[j-1])
                    + (nextRow[j+1] - nextRow[j-1]);

            int gy =  (nextRow[j-1] + 2*nextRow[j] + nextRow[j+1])
                    - (prevRow[j-1] + 2*prevRow[j] + prevRow[j+1]);

            int mag = std::min(255, abs(gx) + abs(gy));
            outRow[j] = (uint8_t)mag;
        }
    }
}

void* worker(void* arg) {
    int id = (intptr_t)arg;

    while (true) {
        pthread_mutex_lock(&mtx);
        while (!work_ready)
            pthread_cond_wait(&cond, &mtx);

        if (stop_all) {
            pthread_mutex_unlock(&mtx);
            break;
        }

        Task& t = tasks[id];
        pthread_mutex_unlock(&mtx);

        if (t.run_gray) {
            for (int r = t.start_row; r < t.end_row; r++) {
                uint8_t* src = t.src->ptr<uint8_t>(r);
                uint8_t* dst = t.dst->ptr<uint8_t>(r);
                for (int c = 0; c < t.src->cols; c+=8) {
                    uint8x8x3_t pix = vld3_u8(src + c*3);
                    uint16x8_t temp = vmull_u8(pix.val[0], vdup_n_u8(29));
                    temp = vmlal_u8(temp, pix.val[1], vdup_n_u8(150));
                    temp = vmlal_u8(temp, pix.val[2], vdup_n_u8(77));
                    vst1_u8(dst + c, vshrn_n_u16(temp, 8));
                }
            }
        }

        if (t.run_sobel) {
            // (use your existing optimized Sobel code here)
            sobelThread(&t);
        }
    }

    return nullptr;
}

