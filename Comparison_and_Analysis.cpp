#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "opencv2/core/parallel/parallel_backend.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

/**
 * @brief 计算图像的宽度和高度缩放因子
 *
 * @param input_width 输入图像的宽度
 * @param input_height 输入图像的高度
 * @param output_width 输出图像的宽度
 * @param output_height 输出图像的高度
 * @param width_scale 输出图像宽度的缩放因子
 * @param height_scale 输出图像高度的缩放因子
 */
void calculateScalingFactors(int input_width, int input_height, int output_width, int output_height,
                             double &width_scale, double &height_scale) {
    width_scale = static_cast<double>(input_width) / output_width;
    height_scale = static_cast<double>(input_height) / output_height;
}

/**
 * @brief 根据输出图像的坐标，通过缩放因子找到输入图像中最近的像素坐标
 *
 * @param x_output 输出图像中的x坐标
 * @param y_output 输出图像中的y坐标
 * @param width_scale 宽度缩放因子
 * @param height_scale 高度缩放因子
 * @return pair<int, int> 输入图像中最近的像素坐标 (x_input, y_input)
 */
pair<int, int> getNearestNeighborCoordinates(int x_output, int y_output, double width_scale, double height_scale) {
    // 计算输入图像中的对应浮点坐标
    double x_input_f = x_output * width_scale;
    double y_input_f = y_output * height_scale;

    // 四舍五入到最近的整数坐标
    int x_input = static_cast<int>(round(x_input_f));
    int y_input = static_cast<int>(round(y_input_f));

    return {x_input, y_input};
}

/**
 * @brief 将坐标限制在输入图像的有效范围内
 *
 * @param x_input 输入图像中的x坐标
 * @param y_input 输入图像中的y坐标
 * @param input_width 输入图像的宽度
 * @param input_height 输入图像的高度
 * @return pair<int, int> 限制后的坐标 (x_clamped, y_clamped)
 */
pair<int, int> clampCoordinates(int x_input, int y_input, int input_width, int input_height) {
    x_input = min(max(x_input, 0), input_width - 1);
    y_input = min(max(y_input, 0), input_height - 1);
    return {x_input, y_input};
}

/**
 * @brief 将输入图像中的像素值赋给输出图像，支持单通道和三通道
 *
 * @param src_image 输入图像
 * @param dst_image 输出图像
 * @param x_output 输出图像中的x坐标
 * @param y_output 输出图像中的y坐标
 * @param x_input 输入图像中的x坐标
 * @param y_input 输入图像中的y坐标
 */
void assignPixelOptimized(const Mat &src_image, Mat &dst_image, int x_output, int y_output, int x_input, int y_input) {
    int channels = src_image.channels();
    if (channels == 1) {
        // 单通道（灰度图像）
        dst_image.at<uchar>(y_output, x_output) = src_image.at<uchar>(y_input, x_input);
    }
    else if (channels == 3) {
        // 三通道（例如RGB/BGR图像）
        Vec3b pixel = src_image.at<Vec3b>(y_input, x_input);
        dst_image.at<Vec3b>(y_output, x_output) = pixel;
    }
    // 可以根据需要扩展更多通道的支持
}

/**
 * @brief 定义一个并行任务，用于在多线程环境下进行图像缩放
 */
struct ResizeTask : public ParallelLoopBody {
    const Mat &src_image;
    Mat &dst_image;
    double width_scale;
    double height_scale;
    int output_height;
    int output_width;

    /**
     * @brief 构造函数
     *
     * @param in 输入图像
     * @param out 输出图像
     * @param sw 宽度缩放因子
     * @param sh 高度缩放因子
     * @param oh 输出图像高度
     * @param ow 输出图像宽度
     */
    ResizeTask(const Mat &in, Mat &out, double sw, double sh, int oh, int ow) :
        src_image(in), dst_image(out), width_scale(sw), height_scale(sh), output_height(oh), output_width(ow) {}

    /**
     * @brief 重载运算符，用于并行处理指定范围的行
     *
     * @param range 需要处理的行范围
     */
    virtual void operator()(const Range &range) const CV_OVERRIDE {
        for (int y_output = range.start; y_output < range.end; ++y_output) {
            for (int x_output = 0; x_output < output_width; ++x_output) {
                // 反向映射到输入图像坐标
                pair<int, int> src_coords = getNearestNeighborCoordinates(x_output, y_output, width_scale, height_scale);
                int x_input = src_coords.first;
                int y_input = src_coords.second;

                // 限制坐标在有效范围内
                pair<int, int> clamped_coords = clampCoordinates(x_input, y_input, src_image.cols, src_image.rows);
                x_input = clamped_coords.first;
                y_input = clamped_coords.second;

                // 赋值像素值到输出图像
                assignPixelOptimized(src_image, dst_image, x_output, y_output, x_input, y_input);
            }
        }
    }
};

/**
 * @brief 使用最近邻插值算法对图像进行缩放，支持多线程处理以优化性能
 *
 * @param input_image 输入的原始图像
 * @param output_width 期望的输出图像宽度
 * @param output_height 期望的输出图像高度
 * @return Mat 缩放后的图像
 */
Mat nearestNeighborResizeParallel(const Mat &input_image, int output_width, int output_height) {
    int input_width = input_image.cols;
    int input_height = input_image.rows;

    double width_scale, height_scale;
    calculateScalingFactors(input_width, input_height, output_width, output_height, width_scale, height_scale);

    // 初始化输出图像
    Mat output_image;
    if (input_image.channels() == 1) {
        output_image = Mat::zeros(output_height, output_width, CV_8UC1);
    }
    else {
        output_image = Mat::zeros(output_height, output_width, input_image.type());
    }

    // 创建并行任务
    ResizeTask task(input_image, output_image, width_scale, height_scale, output_height, output_width);

    // 使用 parallel_for_ 进行多线程处理
    parallel_for_(Range(0, output_height), task);

    return output_image;
}

/**
 * @brief 使用OpenCV的resize函数进行最近邻插值缩放
 *
 * @param input_image 输入的原始图像
 * @param output_width 期望的输出图像宽度
 * @param output_height 期望的输出图像高度
 * @return Mat 缩放后的图像
 */
Mat opencvResize(const Mat &input_image, int output_width, int output_height) {
    Mat resized_image;
    resize(input_image, resized_image, Size(output_width, output_height), 0, 0, INTER_NEAREST);
    return resized_image;
}

/**
 * @brief 主函数，测试并比较自实现的多线程最近邻缩放与OpenCV的性能
 */
int main(int argc, char **argv) {
    // 检查命令行参数
    if (argc < 5) {
        cout << "使用方法: " << argv[0] << " <输入图像路径> <自定义输出图像路径> <输出宽度> <输出高度>" << endl;
        return -1;
    }

    string input_path = argv[1];
    string output_path_custom = argv[2];
    int new_width = stoi(argv[3]);
    int new_height = stoi(argv[4]);

    // 读取输入图像
    Mat input_img = imread(input_path, IMREAD_UNCHANGED);
    if (input_img.empty()) {
        cout << "无法读取图像: " << input_path << endl;
        return -1;
    }

    // 打印图像信息
    cout << "输入图像尺寸: " << input_img.cols << "x" << input_img.rows << endl;
    cout << "图像通道数: " << input_img.channels() << endl;

    // 定义测试参数
    struct TestCase {
        string name;
        int output_width;
        int output_height;
    };

    vector<TestCase> test_cases = {{"Test1_Upscale", input_img.cols * 2, input_img.rows * 2},
                                   {"Test2_Downscale", input_img.cols / 2, input_img.rows / 2},
                                   {"Test3_AspectRatioChange", input_img.cols * 3 / 4, input_img.rows * 5 / 4}};

    // 测试次数以获得平均值
    int test_iterations = 10;

    // 循环测试不同的缩放因子
    for (const auto &test : test_cases) {
        cout << "\n=== " << test.name << " ===" << endl;
        cout << "输出尺寸: " << test.output_width << "x" << test.output_height << endl;

        // 测试自实现的多线程最近邻缩放
        auto start_custom = high_resolution_clock::now();
        Mat resized_img_custom;
        for (int i = 0; i < test_iterations; ++i) {
            resized_img_custom = nearestNeighborResizeParallel(input_img, test.output_width, test.output_height);
        }
        auto end_custom = high_resolution_clock::now();
        auto duration_custom = duration_cast<milliseconds>(end_custom - start_custom).count();
        double avg_time_custom = static_cast<double>(duration_custom) / test_iterations;

        // 测试 OpenCV 的 resize
        auto start_opencv = high_resolution_clock::now();
        Mat resized_img_opencv;
        for (int i = 0; i < test_iterations; ++i) {
            resized_img_opencv = opencvResize(input_img, test.output_width, test.output_height);
        }
        auto end_opencv = high_resolution_clock::now();
        auto duration_opencv = duration_cast<milliseconds>(end_opencv - start_opencv).count();
        double avg_time_opencv = static_cast<double>(duration_opencv) / test_iterations;

        // 保存缩放后的图像
        string custom_output = "custom_" + test.name + ".jpg";
        bool success_custom = imwrite(custom_output, resized_img_custom);
        if (!success_custom) {
            cout << "无法保存自定义缩放图像到: " << custom_output << endl;
            // 继续
        }

        string opencv_output = "opencv_" + test.name + ".jpg";
        bool success_opencv = imwrite(opencv_output, resized_img_opencv);
        if (!success_opencv) {
            cout << "无法保存 OpenCV 缩放图像到: " << opencv_output << endl;
            // 继续
        }

        // 输出比较结果
        cout << "自实现多线程最近邻缩放平均时间: " << avg_time_custom << " ms" << endl;
        cout << "OpenCV 最近邻缩放平均时间: " << avg_time_opencv << " ms" << endl;
    }

    return 0;
}
