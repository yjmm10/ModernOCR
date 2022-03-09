/*
 * @Author: Petrichor
 * @Date: 2022-03-09 15:44:24
 * @LastEditTime: 2022-03-09 17:18:29
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\utils\operators.h
 * 版权声明
 */
#ifndef __OPERATORS_H__
#define __OPERATORS_H__

#include <opencv2/core.hpp>
#include <vector>
namespace op
{
    void SetPadding(cv::Mat &src, cv::Mat &dst, const int padding);
    void ResizeByMaxSide(const cv::Mat &src,cv::Mat &dst,const int max_side_threshold, std::vector<float> &ratio_wh);
} // namespace op


#endif
