#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <stdexcept>
#include <vector>
#include <iostream>

class sobel_mask
{
private:
	size_t width;
	size_t height;

	size_t block_size;

	cv::Mat image;
	cv::Mat img_edges;

public:
	sobel_mask(const std::string& filename, size_t block_size);

	void edge_detection();
};


sobel_mask::sobel_mask(const std::string& filename, size_t block_size = 32)
{
	image = cv::imread(filename);

	if (image.empty())
	{
		throw std::invalid_argument("There is no such file");
	}

	width = image.size().width;
	height = image.size().height;

	if (block_size > width || block_size > height)
	{
		throw std::out_of_range("The given size of block is bigger than both width and height of the image");
	}

	this->block_size = block_size;
}


void sobel_mask::edge_detection()
{
	std::vector<std::vector<cv::Mat>> lines_of_blocks;		//Matice blokù, které pošleme v jednotlivých threadech ke zpracování

	std::vector<cv::Mat> line_of_blocks; //Pomocna promenna
	for (int i = 0; i < (int)(height / block_size); i ++)
	{
		
		for (int j = 0; j < (int)(width / block_size); j ++)
		{
			line_of_blocks.push_back(image(cv::Range(i * block_size, (i + 1) * block_size), cv::Range(j * block_size, (j + 1) * block_size)));
		}
		line_of_blocks.push_back(image(cv::Range(i * block_size, (i + 1) * block_size), cv::Range((int)(width / block_size) * block_size, width)));

		lines_of_blocks.push_back(line_of_blocks);
		line_of_blocks.clear();
	}


	for (int j = 0; j < (int)(width / block_size); j++)
	{
		line_of_blocks.push_back(image(cv::Range((int)(height/block_size) * block_size, height), cv::Range(j * block_size, (j + 1) * block_size)));
	}
	line_of_blocks.push_back(image(cv::Range((int)(height / block_size) * block_size, height), cv::Range((int)(width / block_size) * block_size, width)));
	
	lines_of_blocks.push_back(line_of_blocks);

	line_of_blocks.clear();






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

	cv::imshow("Image", img_edges);

	cv::waitKey(0);

}