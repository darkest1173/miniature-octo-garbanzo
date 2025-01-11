#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * @brief 双线性插值缩放函数
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param target_size 期望的输出图像尺寸
 */
void BilinearInterpolationResizeBasic(const Mat &src, Mat &dst, const Size &target_size) {
    double width_scale = static_cast<double>(src.cols) / target_size.width;
    double height_scale = static_cast<double>(src.rows) / target_size.height;

    dst = Mat::zeros(target_size, src.type());

    for (int x = 0; x < target_size.width; ++x) {
        for (int y = 0; y < target_size.height; ++y) {
            double src_x = (x + 0.5) * width_scale - 0.5;
            double src_y = (y + 0.5) * height_scale - 0.5;
            int x1 = static_cast<int>(src_x);
            int y1 = static_cast<int>(src_y);
            int x2 = min(x1 + 1, src.cols - 1);
            int y2 = min(y1 + 1, src.rows - 1);

            double weight_x2 = src_x - x1;
            double weight_x1 = 1 - weight_x2;
            double weight_y2 = src_y - y1;
            double weight_y1 = 1 - weight_y2;

            if (src.type() == CV_8UC1) {
                double pixel1 = src.at<uchar>(y1, x1);
                double pixel2 = src.at<uchar>(y1, x2);
                double pixel3 = src.at<uchar>(y2, x1);
                double pixel4 = src.at<uchar>(y2, x2);

                double pixel_x1 = weight_x1 * pixel1 + weight_x2 * pixel2;
                double pixel_x2 = weight_x1 * pixel3 + weight_x2 * pixel4;
                double final_pixel = weight_y1 * pixel_x1 + weight_y2 * pixel_x2;

                dst.at<uchar>(y, x) = static_cast<uchar>(final_pixel);
            } else if (src.type() == CV_8UC3) {
                Vec3b pixel1 = src.at<Vec3b>(y1, x1);
                Vec3b pixel2 = src.at<Vec3b>(y1, x2);
                Vec3b pixel3 = src.at<Vec3b>(y2, x1);
                Vec3b pixel4 = src.at<Vec3b>(y2, x2);

                Vec3b pixel_x1 = weight_x1 * pixel1 + weight_x2 * pixel2;
                Vec3b pixel_x2 = weight_x1 * pixel3 + weight_x2 * pixel4;
                Vec3b final_pixel = weight_y1 * pixel_x1 + weight_y2 * pixel_x2;

                dst.at<Vec3b>(y, x) = final_pixel;
            }
        }
    }
}

/**
 * @brief 示例主函数，展示如何使用双线性插值进行图像缩放
 */
int main() {
    // 读取输入图像
    Mat input_image = imread("input_image.jpg", IMREAD_UNCHANGED);
    if (input_image.empty()) {
        cout << "无法读取图像!" << endl;
        return -1;
    }
    // 定义输出尺寸
    int target_width = 800; // 可以调整为任意值，实现上采样或下采样
    int target_height = 600; // 可以调整为任意值，实现上采样或下采样

    // 执行双线性插值缩放
    Mat resized_image;
    BilinearInterpolationResizeBasic(input_image, resized_image, Size(target_width, target_height));

    // 保存缩放后的图像
    imwrite("resized_image_basic.jpg", resized_image);

    // 可选：显示原始和缩放后的图像
    /*
    imshow("原始图像", input_image);
    imshow("缩放后图像 - 基本实现", resized_image);
    waitKey(0);
    */

    cout << "图像缩放完成，保存到 resized_image_basic.jpg" << endl;
    return 0;
}
