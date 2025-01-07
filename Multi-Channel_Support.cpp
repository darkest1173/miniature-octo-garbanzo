#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

/**
 * @brief 最近邻插值算法，可处理单通道(CV_8UC1)或三通道(CV_8UC3)
 * @param src   输入图像 (CV_8UC1 或 CV_8UC3)
 * @param dst   输出图像 (自动分配内存)
 * @param outW  目标图像的宽度
 * @param outH  目标图像的高度
 */
void nearestNeighborInterpolation(const Mat& src, Mat& dst, int outW, int outH)
{
    // 1. 计算缩放比例 (inputWidth / outputWidth, inputHeight / outputHeight)
    double scale_width  = static_cast<double>(src.cols) / static_cast<double>(outW);
    double scale_height = static_cast<double>(src.rows) / static_cast<double>(outH);

    // 2. 初始化目标图像，类型与源图相同
    dst.create(outH, outW, src.type());

    // 3. 获取所需的通道数量及每像素占用的字节数
    int channels = src.channels();           // 1 或 3
    int elemSize = src.elemSize();           // 通常 1*1=1(灰度),或3*1=3(彩色)

    // 4. 遍历目标图像的像素，并进行反向映射
    for(int y_dst = 0; y_dst < outH; y_dst++)
    {
        int y_src = static_cast<int>(std::round(y_dst * scale_height));
        y_src = max(0, min(y_src, src.rows - 1));

        // 获取目标图 y_dst 行的指针
        uchar* dstRowPtr = dst.ptr<uchar>(y_dst);

        for(int x_dst = 0; x_dst < outW; x_dst++)
        {
            int x_src = static_cast<int>(std::round(x_dst * scale_width));
            x_src = max(0, min(x_src, src.cols - 1));

            // 源、目标图像中对应像素的首地址
            const uchar* srcPixelPtr = src.ptr<uchar>(y_src) + x_src * elemSize;
            uchar* dstPixelPtr = dstRowPtr + x_dst * elemSize;

            // 拷贝该像素的全部通道
            for(int c = 0; c < channels; c++)
            {
                dstPixelPtr[c] = srcPixelPtr[c];
            }
        }
    }
}

int main(int argc, char** argv)
{
    // 可从命令行参数获取图像路径，若未指定则默认使用 "input.jpg"
    string inputImage = "input.jpg";
    if(argc > 1)
    {
        inputImage = argv[1];
    }

    // 读取图像，保持原有通道数
    Mat src = imread(inputImage, IMREAD_UNCHANGED);
    if(src.empty())
    {
        cerr << "无法读取图像文件: " << inputImage << endl;
        return -1;
    }

    // 指定目标输出图像的尺寸(宽与高)，可根据需求修改
    int outW = 200;
    int outH = 300;

    // 构建目标图像
    Mat dst;
    nearestNeighborInterpolation(src, dst, outW, outH);

    // 显示结果
    imshow("Source Image", src);
    imshow("NN Interpolation Result", dst);
    waitKey(0);

    // 保存结果到磁盘
    imwrite("nn_output.jpg", dst);

    return 0;
}
