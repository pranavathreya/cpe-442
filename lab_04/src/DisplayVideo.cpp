#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <algorithm>

#define NTHREADS 4

using namespace cv;

void* grayScale(void *args);
void* sobelThread(void *args);

struct matFrames {
    Mat* src;   // for grayscale: BGR frame; for sobel: gray frame
    Mat* dst;   // for grayscale: gray frame; for sobel: sobel output
    int quarter; // 1..4 (row-wise partition)
};

int main(int argc, char** argv)
{
    pthread_t thread_id[NTHREADS];

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

    // Read the frames of the video one by one
    Mat frame;
    while (true) {
        bool ret = cap.read(frame);
        if (!ret) {
            printf("Last frame reached or error.\n");
            break;
        }

        const int rows = frame.rows;
        const int cols = frame.cols;

        Mat gray_frame(rows, cols, CV_8UC1);
        Mat sobel_frame(rows, cols, CV_8UC1, Scalar(0));

        // --- GRAYSCALE (multithreaded like before) ---
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

        // --- SOBEL (implemented the same way: row-wise quarters, pthreads) ---
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

        //imshow("Gray Frame", gray_frame);
        imshow("Sobel Frame", sobel_frame);

        char c = (char)waitKey(25);
        if (c == 'q') break;
    }

    // Clean up
    cap.release();
    destroyAllWindows();
    return 0;
}

void* grayScale(void* args){
    // Cast args to matFrames ptr
    matFrames* mf = static_cast<matFrames*>(args);
    Mat* frame = mf->src;      // BGR input
    Mat* gray_frame = mf->dst; // grayscale output
    int quarter = mf->quarter; // 1..4

    // Apply grayscale conversion using CCIR 601 / OpenCV coefficients
    const int N = gray_frame->rows;
    const int start_row = (quarter - 1) * N / 4;
    const int end_row   = (quarter) * N / 4; // exclusive

    for (int i = start_row; i < end_row; ++i) {
        const Vec3b* frame_i = frame->ptr<Vec3b>(i);
        uint8_t* gray_frame_i = gray_frame->ptr<uint8_t>(i);
        for (int j = 0; j < gray_frame->cols; ++j) {
            uint8_t B = frame_i[j][0];
            uint8_t G = frame_i[j][1];
            uint8_t R = frame_i[j][2];
            gray_frame_i[j] = static_cast<uint8_t>(0.114 * B + 0.587 * G + 0.299 * R);
        }
    }
    return nullptr;
}

void* sobelThread(void* args){
    // Threaded Sobel, mirroring the grayscale threading strategy
    // Each thread processes a contiguous quarter of rows (row-wise partition).
    matFrames* mf = static_cast<matFrames*>(args);
    Mat* gray = mf->src;     // grayscale input
    Mat* sobel = mf->dst;    // sobel output (8-bit)
    int quarter = mf->quarter;

    const int rows = gray->rows;
    const int cols = gray->cols;

    // Compute this quarter's row range [start_row, end_row) and clamp to avoid borders
    int start_row = (quarter - 1) * rows / 4;
    int end_row   = (quarter) * rows / 4; // exclusive

    // Sobel needs neighbors, so skip the first and last rows and columns
    start_row = std::max(1, start_row);
    end_row   = std::min(rows - 1, end_row); // still exclusive

    for (int i = start_row; i < end_row; ++i) {
        const uint8_t* prevRow = gray->ptr<uint8_t>(i - 1);
        const uint8_t* currRow = gray->ptr<uint8_t>(i);
        const uint8_t* nextRow = gray->ptr<uint8_t>(i + 1);
        uint8_t* outRow = sobel->ptr<uint8_t>(i);

        for (int j = 1; j < cols - 1; ++j) {
            int gx = - prevRow[j - 1] + prevRow[j + 1]
                     - 2 * currRow[j - 1] + 2 * currRow[j + 1]
                     - nextRow[j - 1] + nextRow[j + 1];

            int gy = - prevRow[j - 1] - 2 * prevRow[j] - prevRow[j + 1]
                     + nextRow[j - 1] + 2 * nextRow[j] + nextRow[j + 1];

            int mag = std::abs(gx) + std::abs(gy); // L1 magnitude (fast)
            if (mag > 255) mag = 255;
            outRow[j] = static_cast<uint8_t>(mag);
        }
    }

    return nullptr;
}

