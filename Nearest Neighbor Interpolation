#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

/**
 * @brief 使用最近邻插值将图像 src 缩放后赋给 dst
 * @param src  输入图像
 * @param dst  输出图像
 * @param ow   输出图像的宽度
 * @param oh   输出图像的高度
 */
void nearestNeighborInterpolation(const Mat& src, Mat& dst, int ow, int oh)
{
    // 计算缩放比例（x 向右，y 向下）
    double scale_width  = static_cast<double>(src.cols) / static_cast<double>(ow);
    double scale_height = static_cast<double>(src.rows) / static_cast<double>(oh);

    dst.create(oh, ow, src.type());

    // 对输出图像每个像素进行反向映射
    for(int y_dst = 0; y_dst < oh; y_dst++)
    {
        // 根据目标行 y_dst 计算原图行 y_src
        int y_src = static_cast<int>(std::round(y_dst * scale_height));
        // 防止超出边界
        y_src = std::max(0, std::min(y_src, src.rows - 1));

        // 获取目标图 y_dst 行首指针
        uchar* dstRow = dst.ptr<uchar>(y_dst);

        for(int x_dst = 0; x_dst < ow; x_dst++)
        {
            // 根据目标列 x_dst 计算原图列 x_src
            int x_src = static_cast<int>(std::round(x_dst * scale_width));
            x_src = std::max(0, std::min(x_src, src.cols - 1));

            // 获取原图中 (x_src,y_src) 像素指针
            const uchar* srcPixelPtr = src.ptr<uchar>(y_src) + x_src * src.elemSize();
            // 获取目标图 (x_dst,y_dst) 像素指针
            uchar* dstPixelPtr = dstRow + x_dst * src.elemSize();

            // 拷贝像素（包括所有通道）
            for(int c = 0; c < src.elemSize(); c++)
            {
                dstPixelPtr[c] = srcPixelPtr[c];
            }
        }
    }
}

int main()
{
    // 读取原图
    Mat src = imread("input.jpg");
    if(src.empty())
    {
        cout << "无法读取 input.jpg" << endl;
        return -1;
    }

    // 指定输出图像大小
    int outputWidth = 256;
    int outputHeight = 256;

    // 创建目标图像
    Mat dst;
    nearestNeighborInterpolation(src, dst, outputWidth, outputHeight);

    // 显示结果
    imshow("Source Image", src);
    imshow("NN Interpolation Result", dst);
    waitKey(0);

    // 将结果保存到磁盘
    imwrite("output.jpg", dst);

    return 0;
}
