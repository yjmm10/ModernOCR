/*
 * @Author: Petrichor
 * @Date: 2022-03-07 23:34:29
 * @LastEditTime: 2022-03-13 13:03:54
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\core\types.h
 * 版权声明
 */

#ifndef __TYPES_H__
#define __TYPES_H__
#include <opencv2/opencv.hpp>
#include <vector>
namespace ModernOCR{
    #ifdef _WIN32
    #define my_strtol wcstol
    #define my_strrchr wcsrchr
    #define my_strcasecmp _wcsicmp
    #define my_strdup _strdup
    #else
    #define my_strtol strtol
    #define my_strrchr strrchr
    #define my_strcasecmp strcasecmp
    #define my_strdup strdup
    #endif    
    
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
        
        struct TextBlock {
            std::vector<cv::Point> boxPoint;
            float boxScore;
            int angleIndex;
            float angleScore;
            double angleTime;
            std::string text;
            std::vector<float> charScores;
            double crnnTime;
            double blockTime;
        };
        struct OcrResult { 
            double dbNetTime;
            std::vector<TextBlock> textBlocks;
            cv::Mat boxImg;
            double detectTime;
            std::string strRes;
        };

        struct DbNetParam{
            int padding;
            float boxScoreThresh;
            float boxThresh;
            float unClipRatio;
            int maxSideLen;
            
            bool operator==(const DbNetParam& params)
            {
                return (padding == params.padding) && (boxScoreThresh == params.boxScoreThresh) && (boxThresh==params.boxThresh)&&(unClipRatio==params.unClipRatio)&&(maxSideLen==params.maxSideLen);
            };
        };
        struct DetResult {
            std::vector<BoxInfo> allBoxInfo;
            double time;
        };


    };
};
#endif