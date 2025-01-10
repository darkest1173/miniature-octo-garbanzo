#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QProgressBar>
#include <QGridLayout>
#include <QApplication>  // 添加这行
#include <QTextCodec>    // 添加这行
#include <opencv2/opencv.hpp>
#include <QScrollArea>  // 添加到其他include语句后
#include <QtCharts>
#include <QChartView>
#include <QBarSeries>
#include <QBarSet>
#include <QValueAxis>

class ImageResizer : public QMainWindow {
    Q_OBJECT

public:
    explicit ImageResizer(QWidget *parent = nullptr);

protected:
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;

private slots:
    void loadImage();
    void processImage();
    void convertToGray();
    void updateOutputSize(int value);

private:
    void setupUI();
    void displayImage(const cv::Mat& image, QLabel* label);
    void runComparison();
    void createPerformanceChart();
    
    QWidget *centralWidget;
    QLabel *inputImageLabel;
    QLabel *outputImageLabel;
    QLabel *outputImageLabel2;
    QProgressBar *progressBar;
    QPushButton *processButton;
    QPushButton *loadButton;
    QPushButton *convertToGrayButton;
    QComboBox *algorithmSelector;
    QSpinBox *widthSpinBox;
    QSpinBox *heightSpinBox;
    
    cv::Mat inputImage;
    cv::Mat outputImage;
    QString currentImagePath;

    // 图像处理算法
    template <typename T>
    cv::Mat nearestNeighborResizeParallel(const cv::Mat &input, int width, int height);
    template <typename T>
    cv::Mat bilinearResizeParallel(const cv::Mat &input, int width, int height);
    cv::Mat opencvResize(const cv::Mat &input, int width, int height, int interpolation);
    cv::Mat ffmpegResize(const cv::Mat &input, int width, int height, bool isNearest);
    cv::Mat simdResize(const cv::Mat &input, int width, int height);

    // 辅助函数
    void calculateScalingFactors(int input_width, int input_height, int output_width, int output_height,
                               double &scale_width, double &scale_height);
    std::pair<int, int> getNearestNeighborCoordinates(int x_dst, int y_dst, double scale_width, double scale_height);
    std::pair<int, int> clampCoordinates(int x_src, int y_src, int input_width, int input_height);
    
    void runBenchmark();
    void updateResults();
    void displayResults(const std::vector<cv::Mat>& results, const std::vector<double>& times);
    
    // 新增状态变量
    bool isGrayscale = false;
    cv::Mat originalImage;  // 保存原始图像
    std::vector<double> processingTimes;
    std::vector<QString> algorithmNames;
    std::vector<cv::Mat> processedImages;
    
    // UI组件
    QLabel *resultLabel;
    QLabel *timeLabel;
    QTabWidget *resultTabs;
    QScrollArea *inputScrollArea;
    QScrollArea *outputScrollArea;
    QScrollArea *outputScrollArea2;
    QChart *performanceChart;
};
