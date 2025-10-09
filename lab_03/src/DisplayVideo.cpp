#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>

using namespace cv;

int main(int argc, char** argv )
{
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
	while (true) {
		bool ret = cap.read(frame);

		if (!ret) {
			printf("Last frame reached or error.\n");
			break;
		}
		// Apply grayscale conversion into my program flow using CCIR 601 standard
		// OpenCV's COLOR_BGR2GRAY uses the coefficients 0.299, 0.587, and 0.114
		// source: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
		Mat gray_frame;
		cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

		// Apply the sobel filter to single out the edges
		// Use output depth of CV_8U to match the input depth of 8-bit unsigned integers
		Mat sobel_frame;
		cv::Sobel(gray_frame, sobel_frame, CV_8U, 1, 1);

		cv::imshow("Sobel Frame", sobel_frame);
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


Mat to442_grayscale(Mat input){
	int height=input.rows;
	int length=input.cols;
	Mat output(height,length,CV_8U);
	for(int i=0;i<height;i++){
		for(int j=0;j<length;j++){
			printf("%d, %d\n",i,j);
		}
	}	      
       return output;	
}

//Mat to443_sobel(Mat){

//}
