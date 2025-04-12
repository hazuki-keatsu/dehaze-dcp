#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <future>
#include <omp.h>

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
    cv::Mat dark;
    cv::erode(img, dark, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patchSize, patchSize)));
    return dark;
}

// 估计大气光
float estimateAtmosphericLight(const cv::Mat& img, const cv::Mat& dark) {
    int numPixels = img.rows * img.cols;
    int numSamples = std::max(numPixels / 1000, 1); // 取前0.1%的像素

    // 创建像素值-位置对
    std::vector<std::pair<float, int>> pairs(numPixels);
    for (int i = 0; i < numPixels; ++i) {
        pairs[i] = std::make_pair(dark.at<float>(i), i);
    }

    // 使用 partial_sort 找到前 numSamples 个最大值
    std::partial_sort(pairs.begin(), pairs.begin() + numSamples, pairs.end(), std::greater<std::pair<float, int>>());

    // 取最亮像素的平均值
    float sum = 0.0f;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < numSamples; ++i) {
        int idx = pairs[i].second;
        sum += img.at<float>(idx);
    }

    return sum / numSamples;
}

// 估计透射率
cv::Mat estimateTransmission(const cv::Mat& img, float atom, int patchSize, float omega) {
    cv::Mat normalized = img / atom;
    cv::Mat dark = calculateDarkChannel(normalized, patchSize);
    cv::Mat transmission = 1 - omega * dark;
    return transmission;
}

// 恢复无雾图像
cv::Mat recoverScene(const cv::Mat& img, const cv::Mat& transmission, float A, float t0) {
    cv::Mat result(img.size(), CV_32FC1);

    cv::parallel_for_(cv::Range(0, img.rows * img.cols), [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; ++r) {
            int i = r / img.cols;
            int j = r % img.cols;
            float t = std::max(transmission.at<float>(i, j), t0);
            float pixel = (img.at<float>(i, j) - A) / t + A;
            result.at<float>(i, j) = pixel;
        }
        });

    return result;
}

int main() {
    // 启用OpenCL
    cv::ocl::setUseOpenCL(true);

    // 参数设置
    int patchSize = 5;    // 窗口尺寸
    float omega = 0.9f;   // 去雾强度参数
    float t0 = 0.1f;       // 透射率下限

    // 创建计时器
    Timer timer;
    Timer timer2;

    // 读取长红外灰度图像并转换到浮点类型
    cv::Mat img = cv::imread(".\\image\\wuxi_2_0003.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "无法读取图像文件" << std::endl;
        return -1;
    }
    img.convertTo(img, CV_32FC1, 1.0 / 255); // 假设长红外图像是8位

    // 并行计算
    // 计算暗通道
    auto darkFuture = std::async(std::launch::async, calculateDarkChannel, img, patchSize);
    cv::Mat dark = darkFuture.get();

    // 输出程序运行时间
    std::cout << "计算暗通道时间: " << timer.elapsed() << " 毫秒" << std::endl;
    timer.reset();

    // 估计大气光
    auto atomFuture = std::async(std::launch::async, estimateAtmosphericLight, img, dark);
    float atom = atomFuture.get();

    // 输出程序运行时间
    std::cout << "估计大气光时间: " << timer.elapsed() << " 毫秒" << std::endl;
    timer.reset();

    // 估计透射率
    auto transmissionFuture = std::async(std::launch::async, estimateTransmission, img, atom, patchSize, omega);
    cv::Mat transmission = transmissionFuture.get();

    // 输出程序运行时间
    std::cout << "估计透射率时间: " << timer.elapsed() << " 毫秒" << std::endl;
    timer.reset();

    // 恢复无雾图像
    auto resultFuture = std::async(std::launch::async, recoverScene, img, transmission, atom, t0);
    cv::Mat result = resultFuture.get();

    // 输出程序运行时间
    std::cout << "恢复无雾图像时间: " << timer.elapsed() << " 毫秒" << std::endl;

    // 输出程序运行时间
    std::cout << "总时间: " << timer2.elapsed() << " 毫秒" << std::endl;

    // 转换回8位格式并保存
    result.convertTo(result, CV_8UC1, 255);
    cv::imwrite("dehazed_result.jpg", result);

    // 预览图片
    cv::imshow("Display Input", img);
    cv::waitKey(0);
    cv::imshow("Display Result", result);
    cv::waitKey(0);

    cv::waitKey(0);

    return 0;
}
