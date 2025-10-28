#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NTHREADS 4

using namespace cv;

void* grayScale(void *args);
void sobelFilter(Mat* gray_frame, Mat* sobel_frame);

struct matFrames {
		Mat* src;
		Mat* dst;
		int quarter;
};

int main(int argc, char** argv )
{
    pthread_t thread_id[NTHREADS];

    if ( argc != 2 )
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
	int cols, rows;	
	while (true) {
		bool ret = cap.read(frame);

		if (!ret) {
			printf("Last frame reached or error.\n");
			break;
		}

	
		cols = frame.cols, rows = frame.rows;
		Mat gray_frame(rows, cols, CV_8UC1);
		

		// Assign arg struct values
		struct matFrames mf1;
		mf1.src = &frame;
		mf1.dst = &gray_frame;
		mf1.quarter = 1;

		struct matFrames mf2;
		mf2.src = &frame;
		mf2.dst = &gray_frame;
		mf2.quarter = 2;
		
		struct matFrames mf3;
		mf3.src = &frame;
		mf3.dst = &gray_frame;
		mf3.quarter = 3;
		
		struct matFrames mf4;
		mf4.src = &frame;
		mf4.dst = &gray_frame;
		mf4.quarter = 4;
		
		pthread_create( &thread_id[0], NULL, grayScale, (void *) &mf1 );
		pthread_create( &thread_id[1], NULL, grayScale, (void *) &mf2 );
		pthread_create( &thread_id[2], NULL, grayScale, (void *) &mf3 );
		pthread_create( &thread_id[3], NULL, grayScale, (void *) &mf4 );

		for(int j=0; j < 1 /*NTHREADS*/; j++)
		{
			pthread_join( thread_id[j], NULL); 
		}
		cv::imshow("Gray Frame", gray_frame);


		//Mat sobel_frame(rows, cols, CV_8UC1);
		//sobelFilter(&gray_frame,&sobel_frame);
		//cv::imshow("Sobel Frame", sobel_frame);
		//cv::waitKey(0);

		// Wait for 25ms before going to next frame
		// Check if 'q' key is pressed to exit
		char c = (char)cv::waitKey(25);
		if (c == 'q') {
			break;
		}
	}
	// Clean up
	cap.release();
	cv::destroyAllWindows();
	return 0;
}


void* grayScale(void* args){
	// Cast args to matFrames ptr
	matFrames* mf = static_cast<matFrames*>(args);
	Mat* frame = mf->src;
	Mat* gray_frame = mf->dst;
	int quarter = mf->quarter;

	// Apply grayscale conversion into my program flow using CCIR 601 standard
	// OpenCV's COLOR_BGR2GRAY uses the coefficients 0.299, 0.587, and 0.114
	// source: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
	//cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
	int N = gray_frame->rows;
	int start_row = (quarter - 1) * N / 4;
	const Vec3b* frame_i;
	uint8_t* gray_frame_i;
	for (int i=0; i < (N/4); i++) {
		// Point to next row		
		frame_i = frame->ptr<Vec3b>(start_row+i);
		gray_frame_i = gray_frame->ptr<uint8_t>(start_row+i);
		// Vec3b represents a single BGR pixel
		for (int j=0; j < gray_frame->cols; j++) {
			uint8_t B = frame_i[j][0];
			uint8_t G = frame_i[j][1];
			uint8_t R = frame_i[j][2];

			// Apply grayscale:
			// Apparently C++ converts the results of multiplications with
			// floating points to double, hence the typecasting back to uint8_t
			gray_frame_i[j] = static_cast<uint8_t>(0.114 * B + 0.587 * G + 0.299 * R);
		}
	//	printf("start_row: %d, frame_i: %ld\n", start_row, (long) frame_i);
	}

	return gray_frame;
}

void sobelFilter(Mat* gray_frame, Mat* sobel_frame){
// Apply the sobel filter to single out the edges
	// Use output depth of CV_8U to match the input depth of 8-bit unsigned integers
	for (int i=1; i < gray_frame->rows - 1; ++i) {
		const uint8_t* prevRow = gray_frame->ptr<uint8_t>(i - 1);
		const uint8_t* currRow = gray_frame->ptr<uint8_t>(i);
		const uint8_t* nextRow = gray_frame->ptr<uint8_t>(i + 1);
		uint8_t* sobel_frame_i = sobel_frame->ptr<uint8_t>(i);

		for (int j=1; j < gray_frame->cols - 1; ++j) {
			// Use unsigned 16-bits in case of overflow past 0-255
			int16_t Gx = - prevRow[j - 1] + prevRow[j + 1]
				- 2 * currRow[j - 1] + 2 * currRow[j + 1]
				- nextRow[j - 1] + nextRow[j + 1];

			int16_t Gy = -prevRow[j - 1] - 2 * prevRow[j] - prevRow[j + 1]
				+ nextRow[j - 1] + 2 * nextRow[j] + nextRow[j + 1];

			// Use |G| = |Gx| + |Gy| since square and square-root are outrageously
			// expensive
			int16_t absG = abs(Gx) + abs(Gy);

			// Clamp high (no negatives due to abs) values
			// manually to avoid modulo of typecasting
			if (absG > 255) absG = 255;

			// Convert to uint8
			sobel_frame_i[j] = static_cast<uint8_t>(absG);
		}
	}
}
  

