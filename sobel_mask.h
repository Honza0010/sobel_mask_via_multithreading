#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <ostream>

class sobel_mask
{
private:
	size_t width;		//���ka obr�zku
	size_t height;		//v��ka obr�zku

	size_t block_size_w;	//���ka d�l��ho bloku
	size_t block_size_h;	//v��ka d�l��ho bloku

	cv::Mat image;		//Obr�zek
	cv::Mat img_edges;	//Obr�zek hran po detekci

public:
	sobel_mask(const std::string& filename, size_t block_size_h, size_t block_size_w);
	sobel_mask(const std::string& filename);

	void edge_detection_threads();

	void edge_detection();

	void print_edges();

	void my_edge_detection();

private:
	void edge_detection_in_one_piece(cv::Mat& src);

	void resize_image(int new_width, int new_height);		//Tato funkce se vol� v p��pad�, �e obr�zek je moc velk� a nelze jej zobrazit cel�

	void my_sobel(const cv::Mat1b& src, cv::Mat1s& dst, int direction);

};


