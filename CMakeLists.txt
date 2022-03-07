cmake_minimum_required(VERSION 3.14)

project(ocr)


include(cmake/CPM.cmake)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-DUNICODE -D_UNICODE)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions("-Wall -g -O0")
else ()
    add_definitions("-Wall")
endif ()

include_directories(${CMAKE_SOURCE_DIR}/ocr)

add_subdirectory(modules)