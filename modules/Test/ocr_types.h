/*
 * @Author: Petrichor
 * @Date: 2022-03-08 00:05:13
 * @LastEditTime: 2022-03-08 00:23:38
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\Test\ocr_types.h
 * 版权声明
 */
#include<iostream>
namespace ocr
{
    namespace dbnet
    {
        class points
        {
        private:
            /* data */
        public:
            points(/* args */);
            void printt();
            ~points();
        };
        
        points::points(/* args */)
        {
        }
        void points::printt(){
            std::cout<<"printt"<<std::endl;
        }
        points::~points()
        {
        }
        
        
    } // namespace dbnet
    

    
} // namespace ocr


// #endif //__OCR_DBNET_H__