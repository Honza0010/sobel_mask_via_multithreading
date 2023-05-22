#include <iostream>

#include <chrono>


#include "sobel_mask.h"

using namespace cv;
using namespace std;



int main()
{

    try {
        string image_path = "C:/Users/arago/Downloads/opencv_2.png";

        sobel_mask m(image_path);


        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        m.edge_detection_threads();

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Time difference with threads = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

        m.print_edges();

        begin = std::chrono::steady_clock::now();

        m.edge_detection();

        end = std::chrono::steady_clock::now();

        std::cout << "Time difference without threads = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

        m.print_edges();
        //unsigned int number_of_CPU_threads = std::thread::hardware_concurrency();
    }
    catch (exception ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}

