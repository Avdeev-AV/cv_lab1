#include <iostream>
#include <string>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main()
{
	cv::Mat image;


	std::cout << std::endl << "Enter full name of the file: " << std::endl;
	std::string imagename = "";
	std::cin >> imagename;

	try
	{
		image = cv::imread(cv::samples::findFile(imagename), cv::IMREAD_COLOR);
		if (image.empty())
		{
			std::cout << "Image cannot be loaded..!!" << std::endl;
			exit(0);
		}
	}
	catch (cv::Exception)
	{
		std::cout << "Image cannot be loaded..!!" << std::endl;
		exit(0);
	}
	//------------------------------Task 1------------------------------------------
	cv::CascadeClassifier faceDetection;
	if (!faceDetection.load("C:\\Users\\A_And\\source\\repos\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"))
		std::cout << "XML file not found";
	cv::Rect roi_c;
	std::vector<cv::Rect> faces;
	faceDetection.detectMultiScale(image, faces);

	for (int i = 0; i < faces.size(); i++) {
		roi_c.x = faces[i].x;
		roi_c.y = faces[i].y;
		roi_c.width = (faces[i].width);
		roi_c.height = (faces[i].height);
		//-----------------------Green Rectangle--------------------------------------
		//cv::Point pt1(faces[i].x, faces[i].y);
		//cv::Point pt2((faces[i].x + faces[i].height), (faces[i].y + faces[i].width));
		//cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0), 2, 8, 0);

	}
	//------------------------------Task 2------------------------------------------
	const cv::Rect roi(roi_c.x, roi_c.y, roi_c.width + roi_c.width / 10, roi_c.height + roi_c.height / 10);
	image = image(roi).clone();
	//------------------------------Task 3------------------------------------------
	
	//------------------------------Task 4------------------------------------------
	//------------------------------Task 5------------------------------------------
	//------------------------------Task 6------------------------------------------
	//------------------------------Task 7------------------------------------------
	//------------------------------Task 8------------------------------------------
	//------------------------------Task 9------------------------------------------


	cv::namedWindow("Picture", cv::WINDOW_AUTOSIZE);

	imshow("Picture", image);

	cv::waitKey(0);
	cv::destroyAllWindows();

}
