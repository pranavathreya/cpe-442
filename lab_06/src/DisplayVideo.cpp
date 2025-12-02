#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <arm_neon.h>
#include <papi.h>

#define NTHREADS 4

using namespace cv;
using namespace std::chrono;

void* grayScale(void *args);
void* sobelThread(void *args);

struct matFrames {
    Mat* src;   // for grayscale: BGR frame; for sobel: gray frame
    Mat* dst;   // for grayscale: gray frame; for sobel: sobel output
    int quarter; // 1..4 (row-wise partition)
};

int main(int argc, char** argv)
{
	int count=0;
	int start_time=0;
	int stop_time=0;
	float fps=0;

	char c; 
	bool ret;

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

    Mat frame;

    while (true) {
	if(count==0){
			start_time= PAPI_get_real_usec();
	}	
        ret = cap.read(frame);
        if (!ret) {
            printf("Last frame reached or error.\n");
            break;
        }
        
        Mat gray_frame(frame.rows, frame.cols, CV_8UC1);
        Mat sobel_frame(frame.rows, frame.cols, CV_8UC1, Scalar(0));

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

	count++;
	if(count>=10){
		stop_time=PAPI_get_real_usec();
		count=0;
		fps = 10.0/((stop_time-start_time)*pow(10,-6));
		printf("%.2f",fps);
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
    
    c = (char)waitKey(25);
        if (c == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

void* grayScale(void* args){
    matFrames* mf = static_cast<matFrames*>(args);
    Mat* frame = mf->src;      // BGR input
    Mat* gray = mf->dst; // grayscale output
    int quarter = mf->quarter; // 1..4

    const int N = gray->rows;
    const int rows = gray->rows;
    const int cols = gray->cols;
	const int num_pixels= rows*cols;
    const int start_row = (quarter - 1) * N / 4;
    const int end_pixel = num_pixels/4;

    uint8x8x3_t src;
	uint8x8_t       w_r = vdup_n_u8(77);
	uint8x8_t       w_g = vdup_n_u8(150);
	uint8x8_t       w_b = vdup_n_u8(29);
	uint16x8_t      temp;
	uint8x8_t       result;
		uint8_t* currRow = frame->ptr<uint8_t>(start_row);
		uint8_t* currGrayRow = gray->ptr<uint8_t>(start_row);
		for (int j = 0; j < end_pixel; j+=8, currRow += 8 * 3, currGrayRow += 8) {
				src = vld3_u8(currRow);

				temp = vmull_u8(src.val[0], w_b);

				temp = vmlal_u8(temp, src.val[1], w_g);
				temp = vmlal_u8(temp, src.val[2], w_r);

				result = vshrn_n_u16(temp, 8);

				vst1_u8(currGrayRow, result);
		}
	return nullptr;
}

void* sobelThread(void* args) {
    matFrames* mf = static_cast<matFrames*>(args);
    Mat* gray  = mf->src;   // CV_8UC1 input
    Mat* sobel = mf->dst;   // CV_8UC1 output
    int quarter = mf->quarter; // 1..4

    CV_Assert(gray && sobel);
    CV_Assert(gray->type() == CV_8UC1);
    CV_Assert(sobel->type() == CV_8UC1);
    CV_Assert(gray->size() == sobel->size());

    const int rows = gray->rows;
    const int cols = gray->cols;

    int start_row = (quarter - 1) * rows / 4;
    int end_row   = (quarter)     * rows / 4;

    // avoid touching the first/last row (Sobel needs i-1 and i+1)
    start_row = std::max(1, start_row);
    end_row   = std::min(rows - 1, end_row);
	const uint8_t* prevRow; 
	const uint8_t* currRow; 
	const uint8_t* nextRow; 
	uint8_t* outRow;        
    for (int i = start_row; i < end_row; ++i) {
        prevRow = gray->ptr<uint8_t>(i - 1);
        currRow = gray->ptr<uint8_t>(i);
        nextRow = gray->ptr<uint8_t>(i + 1);
        outRow        = sobel->ptr<uint8_t>(i);

        // Vectorized inner loop:
        // j runs from 1..cols-2; for 8-wide blocks that read j-1..j+1,
        // the last valid vector start index is cols-9.
        int j = 1;
        const int j_vec_end = (cols >= 9) ? (cols - 9) : 1; // inclusive end for vector starts
        for (; j <= j_vec_end; j += 8) {
            // Load 8 values for left/center/right from prev, curr, next rows
            uint8x8_t pL = vld1_u8(prevRow + (j - 1));
            uint8x8_t pC = vld1_u8(prevRow + (j    ));
            uint8x8_t pR = vld1_u8(prevRow + (j + 1));

            uint8x8_t cL = vld1_u8(currRow + (j - 1));
            uint8x8_t cR = vld1_u8(currRow + (j + 1));

            uint8x8_t nL = vld1_u8(nextRow + (j - 1));
            uint8x8_t nC = vld1_u8(nextRow + (j    ));
            uint8x8_t nR = vld1_u8(nextRow + (j + 1));

            // Widen to 16-bit signed
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

            // gy = -(pL + 2*pC + pR) + (nL + 2*nC + nR)
            int16x8_t top = vaddq_s16(pL16, pR16);
            top = vaddq_s16(top, vshlq_n_s16(pC16, 1)); // +2*pC
            int16x8_t bot = vaddq_s16(nL16, nR16);
            bot = vaddq_s16(bot, vshlq_n_s16(nC16, 1)); // +2*nC
            int16x8_t gy = vsubq_s16(bot, top);

            // |gx| + |gy| with saturation to 8-bit
            int16x8_t agx = vabsq_s16(gx);
            int16x8_t agy = vabsq_s16(gy);
            int16x8_t mag16 = vqaddq_s16(agx, agy);              // saturate to 16-bit
            uint8x8_t mag8  = vqmovn_u16(vreinterpretq_u16_s16(mag16)); // narrow to 8-bit

            vst1_u8(outRow + j, mag8);
        }
    }
    return nullptr;
}

