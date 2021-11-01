#include <iostream>
#include <string>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

void task_1(cv::Mat image);

int main()
{
	std::cout << std::endl << "Enter full name of the file: " << std::endl;
	std::string imagename = "";
	std::cin >> imagename;

	try
	{
		cv::Mat img = cv::imread(cv::samples::findFile(imagename), cv::IMREAD_COLOR);
		if (img.empty())
		{
			std::cout << "Image cannot be loaded..!!" << std::endl;
			exit(0);
		}

		task_1(img);
	}
	catch (cv::Exception)
	{
		std::cout << "Image cannot be loaded..!!" << std::endl;
		exit(0);
	}


}


void task_1(cv::Mat image)
{
	cv::CascadeClassifier faceDetection;
	if (!faceDetection.load("C:\\Users\\A_And\\source\\repos\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"))
		std::cout << "XML file not found";

	std::vector<cv::Rect> faces;
	faceDetection.detectMultiScale(image, faces);

	for (int i = 0; i < faces.size(); i++) {
		cv::Point pt1(faces[i].x, faces[i].y);
		cv::Point pt2((faces[i].x + faces[i].height + faces[i].height / 10),(faces[i].y + faces[i].width + faces[i].width / 10));
		cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0), 2,8,0);
	}

	cv::namedWindow("original", cv::WINDOW_AUTOSIZE);

	imshow("original", image);

	cv::waitKey(0);
	cv::destroyAllWindows();
}
