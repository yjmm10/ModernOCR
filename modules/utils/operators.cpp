/*
 * @Author: Petrichor
 * @Date: 2022-03-09 15:36:06
 * @LastEditTime: 2022-03-09 17:48:45
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\utils\operators.cpp
 * 版权声明
 */

#include "operators.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
namespace op
{
    void SetPadding(cv::Mat &src, cv::Mat &dst, const int padding) {
        /**
         * @Description:  图像四边添加空白边

         * @Args: 
                src:    源图像
                padding:空白大小

         * @Returns:    目标图像
         */        
        if (padding <= 0)
            src.copyTo(dst); 
        else
            cv::copyMakeBorder(src, dst, padding, padding, padding, padding, cv::BORDER_ISOLATED, cv::Scalar(255, 255, 255));
    }

    void ResizeByMaxSide(const cv::Mat &src,cv::Mat &dst,const int max_side_threshold, std::vector<float> &ratio_wh){
        auto src_w = src.cols;
        auto src_h = src.rows;

        auto ratio = 1.f;
        auto max_side = src_w >= src_h ? src_w : src_h;
        if(max_side > max_side_threshold)
            ratio = src_h > src_w ? float(max_side_threshold/src_h) : float(max_side_threshold/src_w);
        auto dst_h = std::max(int(round(float(src_h*ratio) / 32) * 32), 32);
        auto dst_w = std::max(int(round(float(src_w*ratio) / 32) * 32), 32);
        cv::resize(src,dst,cv::Size(dst_w,dst_h));
        ratio_wh.push_back(float(dst_w)/float(src_w));
        ratio_wh.push_back(float(dst_h)/float(src_h));
    }

    // void ResizeImgType0::Run(const cv::Mat &src, cv::Mat &dst,
    //                      int max_size_len, float &ratio_h, float &ratio_w,
    //                      bool use_tensorrt) {
    //     int w = src.cols;
    //     int h = src.rows;

    //     float ratio = 1.f;
    //     int max_wh = w >= h ? w : h;
    //     if (max_wh > max_size_len) {
    //         if (h > w) {
    //         ratio = float(max_size_len) / float(h);
    //         } else {
    //         ratio = float(max_size_len) / float(w);
    //         }
    //     }

    //     int resize_h = int(float(h) * ratio);
    //     int resize_w = int(float(w) * ratio);

    //     resize_h = max(int(round(float(resize_h) / 32) * 32), 32);
    //     resize_w = max(int(round(float(resize_w) / 32) * 32), 32);

    //     cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
    //     ratio_h = float(resize_h) / float(h);
    //     ratio_w = float(resize_w) / float(w);
    // }

    // cv::Mat resize(cv::Mat &src)
} // namespace op
