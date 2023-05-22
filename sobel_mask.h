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

private:
	void edge_detection_in_one_piece(cv::Mat& src);

	void resize_image(int new_width, int new_height);		//Tato funkce se vol� v p��pad�, �e obr�zek je moc velk� a nelze jej zobrazit cel�
};


sobel_mask::sobel_mask(const std::string& filename, size_t block_size_h, size_t block_size_w)
{
	image = cv::imread(filename);

	if (image.empty())	//Jestli�e se nena�etl ��dn� obr�zek
	{
		throw std::invalid_argument("There is no such file");
	}

	width = image.size().width;
	height = image.size().height;

	if (block_size_w > width || block_size_h > height)	//Byla zad�na velikost d�l��ho bloku v�t�� ne� samotn� obr�zek
	{
		throw std::out_of_range("The given size of block is bigger than both width and height of the image");
	}

	this->block_size_h = block_size_h;
	this->block_size_w = block_size_w;


	if (width > 600)	//P�eveden� na zobrazitelnou velikost
	{
		double h = (double)(600.0 / width);	//Koeficient pro zachov�n� pom�ru obr�zku
		resize_image(600, (int)(h * height));
		this->block_size_h = h*block_size_h;
		this->block_size_w = h*block_size_w;
	}
	if (height > 600)	//P�eveden� na zobrazitelnou velikost
	{
		double w = (double)(600.0 / height);	//Koeficient pro zachov�n� pom�ru obr�zku	
		resize_image((int)(w * width), 600);
		this->block_size_h = w * block_size_h;
		this->block_size_w = w * block_size_w;
	}
}

sobel_mask::sobel_mask(const std::string& filename)
{
	image = cv::imread(filename);

	if (image.empty())	//Jestli�e se nena�etl ��dn� obr�zek
	{
		throw std::invalid_argument("There is no such file");
	}

	width = image.size().width;
	height = image.size().height;


	if (width > 600)	//P�eveden� na zobrazitelnou velikost
	{
		double h = (double)(600.0 / width)*height;
		resize_image(600, (int)h);
	}
	if (height > 600)	//P�eveden� na zobrazitelnou velikost
	{
		double w = (double)(600.0 / height) * width;
		resize_image((int)w, 600);
	}

	this->block_size_h = height/3;
	this->block_size_w = width/3;
}


void sobel_mask::edge_detection()	//Zpracov�n� obr�zku bez tread� jako celku
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
	std::vector<std::vector<cv::Mat>> lines_of_blocks;		//Matice blok�, kter� po�leme v jednotliv�ch threadech ke zpracov�n�. Jedna bu�ka matice je jeden obr�zek ur�en� ke zpracov�n�

	std::vector<cv::Mat> line_of_blocks; //Pomocn� prom�nn�, do kter� budeme ukl�dat jednotliv� ��dky matice a pak to vlo��me na konec

	//Rozd�len� do men��ch obr�zk�
	for (int i = 0; i < (int)(height / block_size_h); i++)	//��dkov� pr�chod
	{
		for (int j = 0; j < (int)(width / block_size_w); j++)	//Sloupcov� pr�chod
		{
			line_of_blocks.push_back(image(cv::Range(i * block_size_h, (i + 1) * block_size_h), cv::Range(j * block_size_w, (j + 1) * block_size_w)));		//Do ��dku v�dy vkl�d�me ��sti obr�zku le��c� vedle sebe
		}
		if (width % block_size_w != 0)	//Pokud ���ka nen� d�liteln� ���kou bloku, tak z�stane n�jak� zbytkov� ��st, kterou mus�me vlo�it tak� do vektoru
		{
			line_of_blocks.push_back(image(cv::Range(i * block_size_h, (i + 1) * block_size_h), cv::Range((int)(width / block_size_w) * block_size_w, width)));
		}
		
		lines_of_blocks.push_back(line_of_blocks);
		line_of_blocks.clear();
	}

	if (height % block_size_h != 0)	//Pokud v��ka nen� d�liten� v��kou bloku, mus�me vlo�it tak� zbytkov� ��dek, kter� bude m�t men�� v��ku
	{
		for (int j = 0; j < (int)(width / block_size_w); j++)
		{
			line_of_blocks.push_back(image(cv::Range((int)(height / block_size_h) * block_size_h, height), cv::Range(j * block_size_w, (j + 1) * block_size_w)));
		}
		if (width % block_size_w != 0)	//Op�t kontrola, zda nen� n�jak� zbytkov� sloupec
		{
			line_of_blocks.push_back(image(cv::Range((int)(height / block_size_h) * block_size_h, height), cv::Range((int)(width / block_size_w) * block_size_w, width)));
		}

		lines_of_blocks.push_back(line_of_blocks);
	}

	line_of_blocks.clear();
	//Konec rozd�len�

	//Thready pro zpracov�n� jednotliv�ch d�lk� obr�zku
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
	//Spojen� ��dkov�ch obr�zk�
	for (int i = 0; i < lines_of_blocks.size(); i++)
	{
		rows_of_edges.push_back(cv::Mat());
		rows_of_edges[i].push_back(lines_of_blocks[i][0]);
		for (int j = 1; j < lines_of_blocks[i].size(); j++)
		{
			cv::hconcat(rows_of_edges[i], lines_of_blocks[i][j], rows_of_edges[i]);		//Spojov�n� ��dkov�ch obr�zk� do jednoho pomoc� funkce hconcat
		}
	}

	//Z�v�re�n� spojen� do p�vodn� velikosti
	for (int i = 0; i < rows_of_edges.size(); i++)
	{
		img_edges.push_back(rows_of_edges[i]);		//Spojov�n� obr�zk� do sloupce pomoc� cv::Mat.push_back
	}

}


void sobel_mask::edge_detection_in_one_piece(cv::Mat& src)	//Funkce pro detekci hran v jednom obr�zku
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


void sobel_mask::print_edges()	//Zobrazen� obr�zku
{
	cv::imshow("Edges", img_edges);

	cv::waitKey(0);	//Po stisknut� esc ukon��
}


void sobel_mask::resize_image(int new_width, int new_height)
{
	cv::resize(image, image, { new_width, new_height }, 0, 0, cv::INTER_NEAREST);	//Zm�n� velikost obr�zku na po�adovanou ���ku a v��ku

	this->width = image.size().width;
	this->height = image.size().height;
}