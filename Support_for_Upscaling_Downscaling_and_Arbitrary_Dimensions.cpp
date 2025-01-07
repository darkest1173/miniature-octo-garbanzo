#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// 使用命名空间以简化代码
using namespace cv;
using namespace std;

/**
 * @brief 使用最近邻插值算法对图像进行缩放
 * 
 * @param input_image 输入的原始图像（单通道或多通道）
 * @param output_width 期望的输出图像宽度
 * @param output_height 期望的输出图像高度
 * @return Mat 缩放后的图像
 */
Mat nearestNeighborResize(const Mat& input_image, int output_width, int output_height) {
    // 获取输入图像的尺寸和通道数
    int input_width = input_image.cols;
    int input_height = input_image.rows;
    int channels = input_image.channels();

    // 计算缩放因子（允许非整数值）
    double scale_width = static_cast<double>(input_width) / output_width;
    double scale_height = static_cast<double>(input_height) / output_height;

    // 初始化输出图像
    Mat output_image(output_height, output_width, input_image.type());

    // 遍历输出图像的每个像素
    for (int y_dst = 0; y_dst < output_height; ++y_dst) {
        for (int x_dst = 0; x_dst < output_width; ++x_dst) {
            // 反向映射：计算对应的输入图像坐标
            double x_src_f = x_dst * scale_width;
            double y_src_f = y_dst * scale_height;

            // 取最近的整数坐标
            int x_src = static_cast<int>(round(x_src_f));
            int y_src = static_cast<int>(round(y_src_f));

            // 确保坐标在输入图像的有效范围内
            x_src = min(max(x_src, 0), input_width - 1);
            y_src = min(max(y_src, 0), input_height - 1);

            // 赋值像素值（处理单通道和多通道）
            if (channels == 1) {
                // 单通道（灰度图像）
                output_image.at<uchar>(y_dst, x_dst) = input_image.at<uchar>(y_src, x_src);
            }
            else {
                // 多通道（例如RGB/BGR图像）
                for (int c = 0; c < channels; ++c) {
                    output_image.at<Vec3b>(y_dst, x_dst)[c] = input_image.at<Vec3b>(y_src, x_src)[c];
                }
            }
        }
    }

    return output_image;
}

int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc != 5) {
        cout << "使用方法: " << argv[0] << " <输入图像路径> <输出图像路径> <输出宽度> <输出高度>" << endl;
        return -1;
    }

    string input_path = argv[1];
    string output_path = argv[2];
    int new_width = stoi(argv[3]);
    int new_height = stoi(argv[4]);

    // 读取输入图像
    Mat input_img = imread(input_path, IMREAD_UNCHANGED);
    if (input_img.empty()) {
        cout << "无法读取图像: " << input_path << endl;
        return -1;
    }

    // 执行最近邻插值缩放
    Mat resized_img = nearestNeighborResize(input_img, new_width, new_height);

    // 保存缩放后的图像
    bool success = imwrite(output_path, resized_img);
    if (!success) {
        cout << "无法保存图像到: " << output_path << endl;
        return -1;
    }

    // 可选：显示原始和缩放后的图像
    /*
    imshow("原始图像", input_img);
    imshow("缩放后图像", resized_img);
    waitKey(0);
    */

    cout << "图像缩放完成，保存到: " << output_path << endl;
    return 0;
}


