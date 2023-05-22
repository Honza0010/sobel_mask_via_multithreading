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
	size_t width;
	size_t height;

	size_t block_size_w;
	size_t block_size_h;

	cv::Mat image;
	cv::Mat img_edges;

public:
	sobel_mask(const std::string& filename, size_t block_size_h, size_t block_size_w);
	sobel_mask(const std::string& filename);

	void edge_detection_threads();

	void edge_detection();

	void print_edges();

private:
	void edge_detection_in_one_piece(cv::Mat& src);
};


sobel_mask::sobel_mask(const std::string& filename, size_t block_size_h, size_t block_size_w)
{
	image = cv::imread(filename);

	if (image.empty())
	{
		throw std::invalid_argument("There is no such file");
	}

	width = image.size().width;
	height = image.size().height;

	if (block_size_w > width || block_size_h > height)
	{
		throw std::out_of_range("The given size of block is bigger than both width and height of the image");
	}

	this->block_size_h = block_size_h;
	this->block_size_w = block_size_w;
}

sobel_mask::sobel_mask(const std::string& filename)
{
	image = cv::imread(filename);

	if (image.empty())
	{
		throw std::invalid_argument("There is no such file");
	}

	width = image.size().width;
	height = image.size().height;

	this->block_size_h = height/3;
	this->block_size_w = width/3;
}


void sobel_mask::edge_detection()
{
	cv::GaussianBlur(image, img_edges, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

	cv::cvtColor(img_edges, img_edges, cv::COLOR_BGR2GRAY);

	cv::Mat grad_x, grad_y;

	cv::Sobel(img_edges, grad_x, CV_8UC1, 1, 0, 5);

	cv::Sobel(img_edges, grad_y, CV_8UC1, 0, 1, 5);

	cv::Mat abs_grad_x, abs_grad_y;
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);

	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, img_edges);
}


void sobel_mask::edge_detection_threads()
{
	std::vector<std::vector<cv::Mat>> lines_of_blocks;		//Matice blokù, které pošleme v jednotlivých threadech ke zpracování

	std::vector<cv::Mat> line_of_blocks; //Pomocna promenna
	for (int i = 0; i < (int)(height / block_size_h); i++)
	{
		for (int j = 0; j < (int)(width / block_size_w); j++)
		{
			line_of_blocks.push_back(image(cv::Range(i * block_size_h, (i + 1) * block_size_h), cv::Range(j * block_size_w, (j + 1) * block_size_w)));
		}
		if (width % block_size_w != 0)
		{
			line_of_blocks.push_back(image(cv::Range(i * block_size_h, (i + 1) * block_size_h), cv::Range((int)(width / block_size_w) * block_size_w, width)));
		}
		
		lines_of_blocks.push_back(line_of_blocks);
		line_of_blocks.clear();
	}

	if (height % block_size_h != 0)
	{
		for (int j = 0; j < (int)(width / block_size_w); j++)
		{
			line_of_blocks.push_back(image(cv::Range((int)(height / block_size_h) * block_size_h, height), cv::Range(j * block_size_w, (j + 1) * block_size_w)));
		}
		if (width % block_size_w != 0)
		{
			line_of_blocks.push_back(image(cv::Range((int)(height / block_size_h) * block_size_h, height), cv::Range((int)(width / block_size_w) * block_size_w, width)));
		}

		lines_of_blocks.push_back(line_of_blocks);
	}

	line_of_blocks.clear();


	//Thready pro zpracovani jednotlivych dilku obrazku
	std::vector<std::thread> threads;
	for (int i = 0; i < lines_of_blocks.size(); i++)
	{
		for (int j = 0; j < lines_of_blocks[i].size(); j++)
		{
			threads.emplace_back(std::thread(&sobel_mask::edge_detection_in_one_piece, this, std::ref(lines_of_blocks[i][j])));
		}
	}

	for (auto& thread : threads)
	{
		thread.join();
	}


	std::vector<cv::Mat> rows_of_edges;


	for (int i = 0; i < lines_of_blocks.size(); i++)
	{
		rows_of_edges.push_back(cv::Mat());
		rows_of_edges[i].push_back(lines_of_blocks[i][0]);
		for (int j = 1; j < lines_of_blocks[i].size(); j++)
		{
			cv::hconcat(rows_of_edges[i], lines_of_blocks[i][j], rows_of_edges[i]);
		}
	}

	for (int i = 0; i < rows_of_edges.size(); i++)
	{
		img_edges.push_back(rows_of_edges[i]);
	}

}


void sobel_mask::edge_detection_in_one_piece(cv::Mat& src)
{
	cv::GaussianBlur(src, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

	cv::Mat grad_x, grad_y;

	cv::Sobel(src, grad_x, CV_8UC1, 1, 0, 5);

	cv::Sobel(src, grad_y, CV_8UC1, 0, 1, 5);

	cv::Mat abs_grad_x, abs_grad_y;
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);

	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src);
}


void sobel_mask::print_edges()
{
	cv::imshow("Edges", img_edges);

	cv::waitKey(0);
}