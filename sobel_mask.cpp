#include "sobel_mask.h"

sobel_mask::sobel_mask(const std::string& filename, size_t block_size_h, size_t block_size_w)
{
	image = cv::imread(filename);

	if (image.empty())	//Jestliže se nenaèetl žádný obrázek
	{
		throw std::invalid_argument("There is no such file");
	}

	width = image.size().width;
	height = image.size().height;

	if (block_size_w > width || block_size_h > height)	//Byla zadána velikost dílèího bloku vìtší než samotný obrázek
	{
		throw std::out_of_range("The given size of block is bigger than both width and height of the image");
	}

	this->block_size_h = block_size_h;
	this->block_size_w = block_size_w;


	if (width > 600)	//Pøevedení na zobrazitelnou velikost
	{
		double h = (double)(600.0 / width);	//Koeficient pro zachování pomìru obrázku
		resize_image(600, (int)(h * height));
		this->block_size_h = h * block_size_h;
		this->block_size_w = h * block_size_w;
	}
	if (height > 600)	//Pøevedení na zobrazitelnou velikost
	{
		double w = (double)(600.0 / height);	//Koeficient pro zachování pomìru obrázku	
		resize_image((int)(w * width), 600);
		this->block_size_h = w * block_size_h;
		this->block_size_w = w * block_size_w;
	}
}

sobel_mask::sobel_mask(const std::string& filename)
{
	image = cv::imread(filename);

	if (image.empty())	//Jestliže se nenaèetl žádný obrázek
	{
		throw std::invalid_argument("There is no such file");
	}

	width = image.size().width;
	height = image.size().height;


	if (width > 600)	//Pøevedení na zobrazitelnou velikost
	{
		double h = (double)(600.0 / width) * height;
		resize_image(600, (int)h);
	}
	if (height > 600)	//Pøevedení na zobrazitelnou velikost
	{
		double w = (double)(600.0 / height) * width;
		resize_image((int)w, 600);
	}

	this->block_size_h = height / 3;
	this->block_size_w = width / 3;
}


void sobel_mask::edge_detection()	//Zpracování obrázku bez treadù jako celku
{
	cv::GaussianBlur(image, img_edges, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

	cv::cvtColor(img_edges, img_edges, cv::COLOR_BGR2GRAY);

	cv::Mat grad_x, grad_y;

	cv::Sobel(img_edges, grad_x, CV_8UC1, 1, 0, 5);		//CV_8UC1 je makro konstanta
	//CV je prefix pro všechna makra v openCv, 8U je 8 bitový unsigned integer, C1 znaèí, že to je grayscale a obrázek má jen jeden kanál
	cv::Sobel(img_edges, grad_y, CV_8UC1, 0, 1, 5);

	cv::Mat abs_grad_x, abs_grad_y;
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);

	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, img_edges);
}


void sobel_mask::edge_detection_threads()
{
	std::vector<std::vector<cv::Mat>> lines_of_blocks;		//Matice blokù, které pošleme v jednotlivých threadech ke zpracování. Jedna buòka matice je jeden obrázek urèení ke zpracování

	std::vector<cv::Mat> line_of_blocks; //Pomocná promìnná, do které budeme ukládat jednotlivé øádky matice a pak to vložíme na konec

	//Rozdìlení do menších obrázkù
	for (int i = 0; i < (int)(height / block_size_h); i++)	//Øádkový prùchod
	{
		for (int j = 0; j < (int)(width / block_size_w); j++)	//Sloupcový prùchod
		{
			line_of_blocks.push_back(image(cv::Range(i * block_size_h, (i + 1) * block_size_h), cv::Range(j * block_size_w, (j + 1) * block_size_w)));		//Do øádku vždy vkládáme èásti obrázku ležící vedle sebe
		}
		if (width % block_size_w != 0)	//Pokud šíøka není dìlitelná šíøkou bloku, tak zùstane nìjaká zbytková èást, kterou musíme vložit také do vektoru
		{
			line_of_blocks.push_back(image(cv::Range(i * block_size_h, (i + 1) * block_size_h), cv::Range((int)(width / block_size_w) * block_size_w, width)));
		}

		lines_of_blocks.push_back(line_of_blocks);
		line_of_blocks.clear();
	}

	if (height % block_size_h != 0)	//Pokud výška není dìlitená výškou bloku, musíme vložit také zbytkový øádek, který bude mít menší výšku
	{
		for (int j = 0; j < (int)(width / block_size_w); j++)
		{
			line_of_blocks.push_back(image(cv::Range((int)(height / block_size_h) * block_size_h, height), cv::Range(j * block_size_w, (j + 1) * block_size_w)));
		}
		if (width % block_size_w != 0)	//Opìt kontrola, zda není nìjaký zbytkový sloupec
		{
			line_of_blocks.push_back(image(cv::Range((int)(height / block_size_h) * block_size_h, height), cv::Range((int)(width / block_size_w) * block_size_w, width)));
		}

		lines_of_blocks.push_back(line_of_blocks);
	}

	line_of_blocks.clear();
	//Konec rozdìlení

	//Thready pro zpracování jednotlivých dílkù obrázku
	std::vector<std::thread> threads;
	for (int i = 0; i < lines_of_blocks.size(); i++)
	{
		for (int j = 0; j < lines_of_blocks[i].size(); j++)
		{
			threads.emplace_back(std::thread(&sobel_mask::edge_detection_in_one_piece, this, std::ref(lines_of_blocks[i][j])));
		}
	}

	for (auto& thread : threads)	//Pøipojení threadù
	{
		thread.join();
	}

	std::vector<cv::Mat> rows_of_edges;
	//Spojení øádkových obrázkù
	for (int i = 0; i < lines_of_blocks.size(); i++)
	{
		rows_of_edges.push_back(cv::Mat());
		rows_of_edges[i].push_back(lines_of_blocks[i][0]);
		for (int j = 1; j < lines_of_blocks[i].size(); j++)
		{
			cv::hconcat(rows_of_edges[i], lines_of_blocks[i][j], rows_of_edges[i]);		//Spojování øádkových obrázkù do jednoho pomocí funkce hconcat
		}
	}

	//Závìreèný spojení do pùvodní velikosti
	for (int i = 0; i < rows_of_edges.size(); i++)
	{
		img_edges.push_back(rows_of_edges[i]);		//Spojování obrázkù do sloupce pomocí cv::Mat.push_back
	}

}


void sobel_mask::edge_detection_in_one_piece(cv::Mat& src)	//Funkce pro detekci hran v jednom obrázku
{
	cv::GaussianBlur(src, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);		//Oèistí obrázek od šumu

	cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);		//Pøevede obrázek na èernobílý

	cv::Mat grad_x, grad_y;

	cv::Sobel(src, grad_x, CV_8UC1, 1, 0, 5);		//Aplikace sobelovy masky ve smìru x

	cv::Sobel(src, grad_y, CV_8UC1, 0, 1, 5);		//Aplikace sobelovy masky ve smìru y

	cv::Mat abs_grad_x, abs_grad_y;
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);

	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src);		//Spojení hran ze smìrù x a y, seète to prvky abs_grad_x a abs_grad_y oba s váhou 0,5 a uloží do matice src
}


void sobel_mask::print_edges()	//Zobrazení obrázku
{
	cv::imshow("Edges", img_edges);

	cv::waitKey(0);	//Po stisknutí esc ukonèí
}


void sobel_mask::resize_image(int new_width, int new_height)
{
	cv::resize(image, image, { new_width, new_height }, 0, 0, cv::INTER_NEAREST);	//Zmìní velikost obrázku na požadovanou šíøku a výšku

	this->width = image.size().width;
	this->height = image.size().height;
}