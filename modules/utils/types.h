/*
 * @Author: Petrichor
 * @Date: 2022-03-07 23:34:29
 * @LastEditTime: 2022-03-11 10:38:45
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\utils\types.h
 * 版权声明
 */

#ifndef __TYPES_H__
#define __TYPES_H__
#include <opencv2/opencv.hpp>
#include<vector>

namespace types{

    struct BoxInfo {
        std::vector<cv::Point> boxPoint;
        float score;
    };

    struct AngleInfo {
        int index;
        float score;
        double time;
    };

    struct RecInfo {
        std::string text;
        std::vector<float> charScores;
        double time;
    };

}

#endif