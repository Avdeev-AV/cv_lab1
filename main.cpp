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
	cv::Mat source_image;

	std::cout << std::endl << "Enter full name of the file: " << std::endl; //lena.jpg tested
	std::string imagename = "";
	std::cin >> imagename;

	try
	{
		source_image = cv::imread(cv::samples::findFile(imagename), cv::IMREAD_COLOR);
		if (source_image.empty())
		{
			std::cout << "Image cannot be loaded!" << std::endl;
			exit(0);
		}
	}
	catch (cv::Exception)
	{
		std::cout << "Image cannot be loaded!" << std::endl;
		exit(0);
	}

	//------------------------------Task 1------------------------------------------

	cv::CascadeClassifier faceDetection;
	if (!faceDetection.load("C:\\Users\\A_And\\source\\repos\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"))
		std::cout << "XML file not found";
	cv::Rect roi_c;
	std::vector<cv::Rect> faces;
	faceDetection.detectMultiScale(source_image, faces);

	for (int i = 0; i < faces.size(); i++) {
		roi_c.x = faces[i].x;
		roi_c.y = faces[i].y;
		roi_c.width = (faces[i].width);
		roi_c.height = (faces[i].height);

		//Green Rectangle Marker
		//cv::Point pt1(faces[i].x, faces[i].y);
		//cv::Point pt2((faces[i].x + faces[i].height), (faces[i].y + faces[i].width));
		//cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0), 2, 8, 0);
	}

	//------------------------------Task 2------------------------------------------

	const cv::Rect roi(roi_c.x, roi_c.y, roi_c.width + roi_c.width / 10, roi_c.height + roi_c.height / 10);
	cv::Mat crop_image = source_image(roi).clone();
	cv::imshow("Face", crop_image);

	//------------------------------Task 3------------------------------------------

	cv::Mat canny_image;
	std::vector<std::vector<cv::Point>> contours, true_contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::Canny(crop_image, canny_image, 50, 100);
	cv::findContours(canny_image, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	//------------------------------Task 4------------------------------------------

	cv::Mat contours_image(canny_image.rows, canny_image.cols, canny_image.type());
	contours_image = cv::Scalar(0,0,0);
	for (const auto& aniter : contours)
	{
		double length = cv::arcLength(aniter, false);
		if (length > 10)
		{
			true_contours.push_back(aniter);
		}
	}
	cv::drawContours(contours_image, true_contours, -1, cv::Scalar(255,255,255),1);
	cv::imshow("Contours", contours_image);

	//------------------------------Task 5------------------------------------------

	cv::Mat morph_image;
	cv::dilate(contours_image, morph_image, cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4)));
	cv::imshow("Morph", morph_image);

	//------------------------------Task 6------------------------------------------

	cv::Mat gaussian_image;
	for (int i = 1; i < 5; i = i + 2)
	{
		cv::GaussianBlur(morph_image, gaussian_image, cv::Size(i, i), 0, 0);
	}
	cv::Mat normalized_image;
	cv::normalize(gaussian_image, normalized_image, 0, 1, 32, CV_32F);

	//------------------------------Task 7------------------------------------------

	cv::Mat bilateral_image;
	for (int i = 1; i < 5; i = i + 2)
	{
		cv::bilateralFilter(crop_image, bilateral_image, i, i * 2, i / 2);
	}
	cv::imshow("Bilateral", bilateral_image);

	//------------------------------Task 8------------------------------------------

	cv::Mat contrast_image,sharp_image, sharpeningKernel = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	cv::filter2D(crop_image, sharp_image, -1, sharpeningKernel);
	sharp_image.convertTo(contrast_image, -1, 2, 0);
	cv::imshow("Sharp", contrast_image);

	//------------------------------Task 9------------------------------------------

	cv::Mat result_image = crop_image.clone();

	for (int x = 0; x < crop_image.rows; ++x)
	{
		for (int y = 0; y < crop_image.cols; ++y)
		{
			for (int c = 0; c < 3; ++c)
			{
				result_image.at<cv::Vec3b>(x, y)[c] = normalized_image.at<float>(x, y) * sharp_image.at<cv::Vec3b>(x, y)[c] + (1.0 - normalized_image.at<float>(x, y)) * bilateral_image.at<cv::Vec3b>(x, y)[c];
			}
		}
	}
	cv::imshow("result", result_image);

	cv::waitKey(0);
	cv::destroyAllWindows();
}
