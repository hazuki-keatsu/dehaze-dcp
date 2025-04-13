#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <future>
#include <omp.h>

// ��ʱ��
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

// ���㰵ͨ��
inline cv::Mat calculateDarkChannel(const cv::Mat& img, int patchSize) {
    cv::Mat dark(img.rows, img.cols, CV_32FC1);

    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    // ����ÿ�����ص�RGB��Сֵ
    cv::min(channels[0], channels[1], dark);
    cv::min(dark, channels[2], dark);

    // ��Сֵ�˲�
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patchSize, patchSize));
    cv::erode(dark, dark, kernel);

    return dark;
}

// ���ƴ�����
inline cv::Vec3f estimateAtmosphericLight(const cv::Mat& img, const cv::Mat& dark) {
    int numPixels = img.rows * img.cols;
    int numSamples = std::max(numPixels / 1000, 1); // ȡǰ0.1%������

    // ��������ֵ-λ�ö�
    std::vector<std::pair<float, int>> pairs(numPixels);
    for (int i = 0; i < numPixels; ++i) {
        pairs[i] = std::make_pair(dark.at<float>(i), i);
    }

    // ʹ�� nth_element �ҵ�ǰ numSamples �����ֵ
    std::nth_element(pairs.begin(), pairs.begin() + numSamples, pairs.end(), std::greater<std::pair<float, int>>());

    // ȡ�������ص�ƽ��ֵ
    cv::Vec3f sum(0, 0, 0);

    cv::parallel_for_(cv::Range(0, numSamples), [&](const cv::Range& range) {
        cv::Vec3f localSum(0, 0, 0);
        for (int i = range.start; i < range.end; ++i) {
            int idx = pairs[i].second;
            localSum += img.at<cv::Vec3f>(idx);
        }
#pragma omp critical
        sum += localSum;
        });

    return sum / numSamples;
}

// ����͸����
inline cv::Mat estimateTransmission(const cv::Mat& img, const cv::Vec3f& atom, int patchSize, float omega) {
    cv::Mat normalized(img.size(), CV_32FC3);

    // ��һ��ͼ��
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            cv::Vec3f pixel = img.at<cv::Vec3f>(i, j);
            normalized.at<cv::Vec3f>(i, j) = cv::Vec3f(pixel[0] / atom[0], pixel[1] / atom[1], pixel[2] / atom[2]);
        }
    }

    // �����һ��ͼ��İ�ͨ��
    cv::Mat dark = calculateDarkChannel(normalized, patchSize);

    // ����͸����
    cv::Mat transmission = 1 - omega * dark;
    return transmission;
}

// �ָ�����ͼ��
inline cv::Mat recoverScene(const cv::Mat& img, const cv::Mat& transmission, const cv::Vec3f& A, float t0) {
    cv::Mat result(img.size(), CV_32FC3);

    cv::parallel_for_(cv::Range(0, img.rows * img.cols), [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; ++r) {
            int i = r / img.cols;
            int j = r % img.cols;
            float t = std::max(transmission.at<float>(i, j), t0);
            cv::Vec3f pixel = (img.at<cv::Vec3f>(i, j) - A) / t + A;
            result.at<cv::Vec3f>(i, j) = pixel;
        }
        });

    return result;
}

int main() {
    // ����OpenCL
    cv::ocl::setUseOpenCL(true);

    // ��������
    int patchSize = 15;    // ���ڳߴ�
    float omega = 0.95f;   // ȥ��ǿ�Ȳ���
    float t0 = 0.1f;       // ͸��������

    // ��ȡͼ��ת������������
    cv::Mat img = cv::imread(".\\images\\wuxi_2_0000.jpg");
    img.convertTo(img, CV_32FC3, 1.0 / 255);

    // ������ʱ��
    Timer timer;
    Timer timer2;

    // ���м���
    // ���㰵ͨ��
    auto darkFuture = std::async(std::launch::async, calculateDarkChannel, img, patchSize);
    cv::Mat dark = darkFuture.get();

    // �����������ʱ��
    std::cout << "���㰵ͨ��ʱ��: " << timer.elapsed() << " ����" << std::endl;
    timer.reset();

    // ���ƴ�����
    auto atomFuture = std::async(std::launch::async, estimateAtmosphericLight, img, dark);
    cv::Vec3f atom = atomFuture.get();

    // �����������ʱ��
    std::cout << "���ƴ�����ʱ��: " << timer.elapsed() << " ����" << std::endl;
    timer.reset();

    // ����͸����
    auto transmissionFuture = std::async(std::launch::async, estimateTransmission, img, atom, patchSize, omega);
    cv::Mat transmission = transmissionFuture.get();

    // �����������ʱ��
    std::cout << "����͸����ʱ��: " << timer.elapsed() << " ����" << std::endl;
    timer.reset();

    // �ָ�����ͼ��
    auto resultFuture = std::async(std::launch::async, recoverScene, img, transmission, atom, t0);
    cv::Mat result = resultFuture.get();

    // �����������ʱ��
    std::cout << "�ָ�����ͼ��ʱ��: " << timer.elapsed() << " ����" << std::endl;

    // �����������ʱ��
    std::cout << "��ʱ��: " << timer2.elapsed() << " ����" << std::endl;

    // ת����8λ��ʽ������
    result.convertTo(result, CV_8UC3, 255);
    cv::imwrite("dehazed_result.jpg", result);

    // Ԥ��ͼƬ
    cv::imshow("Display Input", img);
    cv::waitKey(0);
    cv::imshow("Display Result", result);
    cv::waitKey(0);

    cv::waitKey(0);

    return 0;
}
