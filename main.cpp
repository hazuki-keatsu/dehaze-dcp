#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

// 计时器
#include <chrono>
class Timer {
public:
    Timer() : start_time_point(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time_point = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        auto end_time_point = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time_point - start_time_point).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_point;
};

// 计算暗通道
cv::Mat calculateDarkChannel(const cv::Mat& img, int patchSize) {
    cv::Mat dark = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // 计算每个像素的RGB最小值
    cv::min(channels[0], channels[1], dark);
    cv::min(dark, channels[2], dark);

    // 最小值滤波
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patchSize, patchSize));
    cv::erode(dark, dark, kernel);

    return dark;
}

// 估计大气光
cv::Vec3f estimateAtmosphericLight(const cv::Mat& img, const cv::Mat& dark) {
    int numPixels = img.rows * img.cols;
    int numSamples = std::max(numPixels / 1000, 1); // 取前0.1%的像素

    // 创建像素值-位置对
    std::vector<std::pair<float, int>> pairs(numPixels);
    for (int i = 0; i < numPixels; ++i) {
        int row = i / img.cols;
        int col = i % img.cols;
        pairs[i] = std::make_pair(dark.at<float>(row, col), i);
    }

    // 按暗通道值降序排序
    std::sort(pairs.begin(), pairs.end(), std::greater<std::pair<float, int>>());

    // 取最亮像素的平均值
    cv::Vec3f sum(0, 0, 0);
    for (int i = 0; i < numSamples; ++i) {
        int idx = pairs[i].second;
        int row = idx / img.cols;
        int col = idx % img.cols;
        sum += img.at<cv::Vec3f>(row, col);
    }

    return sum / numSamples;
}

// 估计透射率
cv::Mat estimateTransmission(const cv::Mat& img, const cv::Vec3f& atom, int patchSize, float omega) {
    cv::Mat normalized(img.size(), CV_32FC3);

    // 归一化图像
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            cv::Vec3f pixel = img.at<cv::Vec3f>(i, j);
            normalized.at<cv::Vec3f>(i, j) = cv::Vec3f(pixel[0] / atom[0], pixel[1] / atom[1], pixel[2] / atom[2]);
        }
    }

    // 计算归一化图像的暗通道
    cv::Mat dark = calculateDarkChannel(normalized, patchSize);

    // 计算透射率
    cv::Mat transmission = 1 - omega * dark;
    return transmission;
}

// 恢复无雾图像
cv::Mat recoverScene(const cv::Mat& img, const cv::Mat& transmission, const cv::Vec3f& A, float t0) {
    cv::Mat result(img.size(), CV_32FC3);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            float t = std::max(transmission.at<float>(i, j), t0);
            cv::Vec3f pixel = (img.at<cv::Vec3f>(i, j) - A) / t + A;
            result.at<cv::Vec3f>(i, j) = pixel;
        }
    }

    return result;
}

int main() {
    // 参数设置
    int patchSize = 15;    // 窗口尺寸
    float omega = 0.95f;   // 去雾强度参数
    float t0 = 0.1f;       // 透射率下限

    // 创建计时器
    Timer timer;

    // 读取图像并转换到浮点类型
    cv::Mat img = cv::imread(".\\image\\tiananmen.png");
    img.convertTo(img, CV_32FC3, 1.0 / 255);

    // 计算暗通道
    cv::Mat dark = calculateDarkChannel(img, patchSize);

    // 估计大气光
    cv::Vec3f atom = estimateAtmosphericLight(img, dark);

    // 估计透射率
    cv::Mat transmission = estimateTransmission(img, atom, patchSize, omega);

    // 恢复无雾图像
    cv::Mat result = recoverScene(img, transmission, atom, t0);

    // 输出程序运行时间
    std::cout << "程序运行时间: " << timer.elapsed() << " 毫秒" << std::endl;

    // 转换回8位格式并保存
    result.convertTo(result, CV_8UC3, 255);
    cv::imwrite("dehazed_result.jpg", result);

    // 预览图片
    cv::imshow("Display Input", img);
    cv::waitKey(0);
    cv::imshow("Display Result", result);
    cv::waitKey(0);

    cv::waitKey(0);

    return 0;
}
