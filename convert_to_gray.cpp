#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "用法: " << argv[0] << " <输入图像路径> <输出图像路径>" << endl;
        return -1;
    }

    // 读取输入图像
    Mat input_image = imread(argv[1]);
    if (input_image.empty()) {
        cerr << "无法读取图像: " << argv[1] << endl;
        return -1;
    }

    // 转换为CV_8UC1
    Mat gray_image;
    if (input_image.type() != CV_8UC1) {
        if (input_image.channels() > 1) {
            cvtColor(input_image, gray_image, COLOR_BGR2GRAY);
        } else {
            input_image.convertTo(gray_image, CV_8UC1);
        }
    } else {
        gray_image = input_image.clone();
    }

    // 验证转换结果
    if (gray_image.type() != CV_8UC1) {
        cerr << "转换失败！" << endl;
        return -1;
    }

    // 保存结果
    if (!imwrite(argv[2], gray_image)) {
        cerr << "无法保存图像: " << argv[2] << endl;
        return -1;
    }

    cout << "转换完成！输出图像类型: CV_8UC1" << endl;
    cout << "图像尺寸: " << gray_image.size() << endl;
    return 0;
}