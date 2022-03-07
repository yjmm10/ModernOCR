cmake_minimum_required(VERSION 3.14)

project(utils VERSION 0.1.0)


# # OnnxRuntime
# include(${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime-static/OnnxRuntimeWrapper.cmake)

# find_package(OnnxRuntime REQUIRED)
# if (OnnxRuntime_FOUND)
#     message(STATUS "OnnxRuntime_LIBS: ${OnnxRuntime_LIBS}")
#     message(STATUS "OnnxRuntime_INCLUDE_DIRS: ${OnnxRuntime_INCLUDE_DIRS}")
# else ()
#     message(FATAL_ERROR "onnxruntime Not Found!")
# endif (OnnxRuntime_FOUND)

# # OpenCV
# set(BUILD_SHARED_LIBS false)
# include(${CMAKE_SOURCE_DIR}/3rdparty/opencv-static/OpenCVWrapperConfig.cmake)
# find_package(OpenCV REQUIRED)
# if (OpenCV_FOUND)
#     message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
#     message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
# else ()
#     message(FATAL_ERROR "opencv Not Found!")
# endif (OpenCV_FOUND)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)

add_library(${PROJECT_NAME} ${SRCS})

target_link_libraries(${PROJECT_NAME} 
    PUBLIC ${OpenCV_LIBS} )

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


