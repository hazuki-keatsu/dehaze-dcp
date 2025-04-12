#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

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
cv::Mat calculateDarkChannel(const cv::Mat& img, int patchSize) {
    cv::Mat dark = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
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
cv::Vec3f estimateAtmosphericLight(const cv::Mat& img, const cv::Mat& dark) {
    int numPixels = img.rows * img.cols;
    int numSamples = std::max(numPixels / 1000, 1); // ȡǰ0.1%������

    // ��������ֵ-λ�ö�
    std::vector<std::pair<float, int>> pairs(numPixels);
    for (int i = 0; i < numPixels; ++i) {
        int row = i / img.cols;
        int col = i % img.cols;
        pairs[i] = std::make_pair(dark.at<float>(row, col), i);
    }

    // ����ͨ��ֵ��������
    std::sort(pairs.begin(), pairs.end(), std::greater<std::pair<float, int>>());

    // ȡ�������ص�ƽ��ֵ
    cv::Vec3f sum(0, 0, 0);
    for (int i = 0; i < numSamples; ++i) {
        int idx = pairs[i].second;
        int row = idx / img.cols;
        int col = idx % img.cols;
        sum += img.at<cv::Vec3f>(row, col);
    }

    return sum / numSamples;
}

// ����͸����
cv::Mat estimateTransmission(const cv::Mat& img, const cv::Vec3f& atom, int patchSize, float omega) {
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
    // ��������
    int patchSize = 15;    // ���ڳߴ�
    float omega = 0.95f;   // ȥ��ǿ�Ȳ���
    float t0 = 0.1f;       // ͸��������

    // ������ʱ��
    Timer timer;

    // ��ȡͼ��ת������������
    cv::Mat img = cv::imread(".\\image\\tiananmen.png");
    img.convertTo(img, CV_32FC3, 1.0 / 255);

    // ���㰵ͨ��
    cv::Mat dark = calculateDarkChannel(img, patchSize);

    // ���ƴ�����
    cv::Vec3f atom = estimateAtmosphericLight(img, dark);

    // ����͸����
    cv::Mat transmission = estimateTransmission(img, atom, patchSize, omega);

    // �ָ�����ͼ��
    cv::Mat result = recoverScene(img, transmission, atom, t0);

    // �����������ʱ��
    std::cout << "��������ʱ��: " << timer.elapsed() << " ����" << std::endl;

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
