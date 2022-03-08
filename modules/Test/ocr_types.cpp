#include<iostream>
// #include<ocr_types.h>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
// #DEFINE LOG(x)() log

#define varName(x) #x
using namespace spdlog;
class A{
    public:
    A(){
        logger = spdlog::get("DbNet");
        if(logger==nullptr)
        {
            logger = spdlog::basic_logger_mt("DbNet", "logs/ModernOCR.txt");
            logger->info("Create DbNet logs!");
        }else{
            logger->info("Load DbNet logs!");
        }
    };
    ~A(){};
    
    void p(){
        logger->info("wedo");
        logger->warn("wedo你们");
        };
        std::string getClassName() {
            return typeid(*this).name();
        }
    private:
        std::shared_ptr<spdlog::logger> logger;
};
// #include <typeinfo.h>
 


int main(int argc, char const *argv[])
{
    auto my_logger = spdlog::basic_logger_mt("file_logger", "logs/basic-log.txt");
    
    auto my_logger2 = spdlog::basic_logger_mt("happy", "logs/basic-log.txt");
    A a;
    // a.log = my_logger;
    a.p();
    my_logger2->info("dfklsdjfldksyoudnkn{}", varName(a));
    my_logger2->info("dfklsdjfldksyoudnkn{}",varName(my_logger));
    // my_logger2->info("dfklsdjfldksyoudnkn{}",a.getClassName());
    auto hhh = spdlog::get("happy");
    hhh->info("winlema");
    // /* code */
    // ocr::dbnet::points a;
    // a.printt();
    // return 0;
}
