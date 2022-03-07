cmake_minimum_required(VERSION 3.14)

project(ocr)

include(cmake/CPM.cmake)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set(CMAKE_CXX_STANDARD 11)
add_definitions(-DUNICODE -D_UNICODE)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions("-Wall -g -O0")
else ()
    add_definitions("-Wall")
endif ()
# 去除提醒
add_definitions(-w)

include_directories(${CMAKE_SOURCE_DIR}/modules)

add_subdirectory(modules)