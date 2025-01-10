#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp> // 用于 Universal Intrinsics

using namespace std;
using namespace cv;

// 计算缩放因子
void calculateScalingFactors(int input_width, int input_height, int output_width, int output_height,
                             double &scale_width, double &scale_height) {
    scale_width = static_cast<double>(input_width) / output_width;
    scale_height = static_cast<double>(input_height) / output_height;
}

// 最近邻插值源坐标
inline pair<int, int> getNearestNeighborCoordinates(int x_dst, int y_dst, double scale_width, double scale_height) {
    int x_src = static_cast<int>(x_dst * scale_width + 0.5);
    int y_src = static_cast<int>(y_dst * scale_height + 0.5);
    return {x_src, y_src};
}

// 限制坐标在有效范围内
inline pair<int, int> clampCoordinates(int x_src, int y_src, int input_width, int input_height) {
    x_src = min(max(x_src, 0), input_width - 1);
    y_src = min(max(y_src, 0), input_height - 1);
    return {x_src, y_src};
}

// 多数据类型支持的像素赋值函数
template <typename T>
inline void assignPixel(const Mat &input_image, Mat &output_image, int x_dst, int y_dst, int x_src, int y_src) {
    int channels = input_image.channels();
    if (channels == 1) {
        output_image.at<T>(y_dst, x_dst) = input_image.at<T>(y_src, x_src);
    } else if (channels == 3) {
        Vec<T, 3> pixel = input_image.at<Vec<T, 3>>(y_src, x_src);
        output_image.at<Vec<T, 3>>(y_dst, x_dst) = pixel;
    }
}

// 最近邻插值缩放（多线程，多数据类型）
template <typename T>
Mat nearestNeighborResizeParallel(const Mat &input_image, int output_width, int output_height) {
    int input_width = input_image.cols;
    int input_height = input_image.rows;

    double scale_width, scale_height;
    calculateScalingFactors(input_width, input_height, output_width, output_height, scale_width, scale_height);

    Mat output_image(output_height, output_width, input_image.type());

    parallel_for_(Range(0, output_height), [&](const Range &range) {
        for (int y_dst = range.start; y_dst < range.end; ++y_dst) {
            int y_src = static_cast<int>(y_dst * scale_height + 0.5);
            y_src = min(max(y_src, 0), input_height - 1);

            T *output_row = output_image.ptr<T>(y_dst);
            for (int x_dst = 0; x_dst < output_width; ++x_dst) {
                int x_src = static_cast<int>(x_dst * scale_width + 0.5);
                x_src = min(max(x_src, 0), input_width - 1);

                if (input_image.channels() == 1) {
                    output_row[x_dst] = input_image.at<T>(y_src, x_src);
                } else if (input_image.channels() == 3) {
                    Vec<T, 3> pixel = input_image.at<Vec<T, 3>>(y_src, x_src);
                    output_image.at<Vec<T, 3>>(y_dst, x_dst) = pixel;
                }
            }
        }
    });

    return output_image;
}

// 双线性插值缩放（多线程，多数据类型）
template <typename T>
Mat bilinearResizeParallel(const Mat &input_image, int output_width, int output_height) {
    int input_width = input_image.cols;
    int input_height = input_image.rows;

    double scale_width = static_cast<double>(input_width) / output_width;
    double scale_height = static_cast<double>(input_height) / output_height;

    Mat output_image(output_height, output_width, input_image.type());

    parallel_for_(Range(0, output_height), [&](const Range &range) {
        for (int y_dst = range.start; y_dst < range.end; ++y_dst) {
            double y_src = (y_dst + 0.5) * scale_height - 0.5;
            int y0 = static_cast<int>(floor(y_src));
            int y1 = y0 + 1;
            double y_lerp = y_src - y0;

            y0 = min(max(y0, 0), input_height - 1);
            y1 = min(max(y1, 0), input_height - 1);

            for (int x_dst = 0; x_dst < output_width; ++x_dst) {
                double x_src = (x_dst + 0.5) * scale_width - 0.5;
                int x0 = static_cast<int>(floor(x_src));
                int x1 = x0 + 1;
                double x_lerp = x_src - x0;

                x0 = min(max(x0, 0), input_width - 1);
                x1 = min(max(x1, 0), input_width - 1);

                if (input_image.channels() == 1) {
                    T p00 = input_image.at<T>(y0, x0);
                    T p01 = input_image.at<T>(y0, x1);
                    T p10 = input_image.at<T>(y1, x0);
                    T p11 = input_image.at<T>(y1, x1);

                    double value = (1 - y_lerp) * ((1 - x_lerp) * p00 + x_lerp * p01) +
                                   y_lerp * ((1 - x_lerp) * p10 + x_lerp * p11);

                    if (std::is_same<T, uchar>::value || std::is_same<T, uint16_t>::value) {
                        output_image.at<T>(y_dst, x_dst) = static_cast<T>(round(value));
                    } else {
                        output_image.at<T>(y_dst, x_dst) = static_cast<T>(value);
                    }
                } else if (input_image.channels() == 3) {
                    Vec<T, 3> p00 = input_image.at<Vec<T, 3>>(y0, x0);
                    Vec<T, 3> p01 = input_image.at<Vec<T, 3>>(y0, x1);
                    Vec<T, 3> p10 = input_image.at<Vec<T, 3>>(y1, x0);
                    Vec<T, 3> p11 = input_image.at<Vec<T, 3>>(y1, x1);

                    Vec<double, 3> value;
                    for (int c = 0; c < 3; ++c) {
                        value[c] = (1 - y_lerp) * ((1 - x_lerp) * p00[c] + x_lerp * p01[c]) +
                                   y_lerp * ((1 - x_lerp) * p10[c] + x_lerp * p11[c]);

                        if (std::is_same<T, uchar>::value || std::is_same<T, uint16_t>::value) {
                            output_image.at<Vec<T, 3>>(y_dst, x_dst)[c] = static_cast<T>(round(value[c]));
                        } else {
                            output_image.at<Vec<T, 3>>(y_dst, x_dst)[c] = static_cast<T>(value[c]);
                        }
                    }
                }
            }
        }
    });

    return output_image;
}

// 使用 Universal Intrinsics 的最近邻插值 SIMD 加速
void nearestNeighborResizeSIMD(const Mat &input_image, Mat &output_image, double scale_width, double scale_height) {
    int input_width = input_image.cols;
    int input_height = input_image.rows;
    int output_width = output_image.cols;
    int output_height = output_image.rows;

    // 仅支持单通道 8 位图像的 SIMD 优化
    if (input_image.type() == CV_8UC1) {
        int vec_size = v_uint8::nlanes;

        parallel_for_(Range(0, output_height), [&](const Range &range) {
            for (int y_dst = range.start; y_dst < range.end; ++y_dst) {
                uchar *output_row = output_image.ptr<uchar>(y_dst);
                int y_src = static_cast<int>(y_dst * scale_height + 0.5);
                y_src = min(max(y_src, 0), input_height - 1);
                const uchar *input_row = input_image.ptr<uchar>(y_src);

                int x_dst = 0;
                for (; x_dst <= output_width - vec_size; x_dst += vec_size) {
                    int x_src_indices[vec_size];
                    for (int i = 0; i < vec_size; ++i) {
                        int x_src = static_cast<int>((x_dst + i) * scale_width + 0.5);
                        x_src_indices[i] = min(max(x_src, 0), input_width - 1);
                    }
                    v_uint8 pixels = v_lut(input_row, x_src_indices);
                    v_store(output_row + x_dst, pixels);
                }
                // 处理剩余部分
                for (; x_dst < output_width; ++x_dst) {
                    int x_src = static_cast<int>(x_dst * scale_width + 0.5);
                    x_src = min(max(x_src, 0), input_width - 1);
                    output_row[x_dst] = input_row[x_src];
                }
            }
        });
    } else {
        // 其他类型使用普通的最近邻插值
        output_image = nearestNeighborResizeParallel<uchar>(input_image, output_width, output_height);
    }
}

int main() {
    // 读取图像，支持任意类型
    Mat input_image = imread("input_image.png", IMREAD_UNCHANGED);
    if (input_image.empty()) {
        cerr << "无法读取图像！" << endl;
        return -1;
    }

    int new_width = input_image.cols * 2;  // 设置新的宽度
    int new_height = input_image.rows * 2; // 设置新的高度

    Mat output_image;

    int type = input_image.type();
    switch (type) {
        case CV_8UC1:
        case CV_8UC3:
            output_image = nearestNeighborResizeParallel<uchar>(input_image, new_width, new_height);
            break;
        case CV_16UC1:
        case CV_16UC3:
            output_image = nearestNeighborResizeParallel<uint16_t>(input_image, new_width, new_height);
            break;
        case CV_32FC1:
        case CV_32FC3:
            output_image = nearestNeighborResizeParallel<float>(input_image, new_width, new_height);
            break;
        default:
            cerr << "不支持的图像类型！" << endl;
            return -1;
    }

    // 保存最近邻插值结果
    imwrite("output_nearest.png", output_image);

    // 双线性插值
    Mat bilinear_output_image;
    switch (type) {
        case CV_8UC1:
        case CV_8UC3:
            bilinear_output_image = bilinearResizeParallel<uchar>(input_image, new_width, new_height);
            break;
        case CV_16UC1:
        case CV_16UC3:
            bilinear_output_image = bilinearResizeParallel<uint16_t>(input_image, new_width, new_height);
            break;
        case CV_32FC1:
        case CV_32FC3:
            bilinear_output_image = bilinearResizeParallel<float>(input_image, new_width, new_height);
            break;
        default:
            cerr << "不支持的图像类型！" << endl;
            return -1;
    }

    // 保存双线性插值结果
    imwrite("output_bilinear.png", bilinear_output_image);

    // 使用 SIMD 加速的最近邻插值（仅支持 CV_8UC1）
    if (input_image.type() == CV_8UC1) {
        Mat simd_output_image(new_height, new_width, input_image.type());
        double scale_width = static_cast<double>(input_image.cols) / new_width;
        double scale_height = static_cast<double>(input_image.rows) / new_height;
        nearestNeighborResizeSIMD(input_image, simd_output_image, scale_width, scale_height);
        imwrite("output_simd.png", simd_output_image);
    }

    cout << "图像缩放完成！" << endl;

    return 0;
}
