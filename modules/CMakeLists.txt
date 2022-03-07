cmake_minimum_required(VERSION 3.14)

project(modules VERSION 0.1.0)

# PackageProject
CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.8.0")

# OpenCV
CPMaddPackage(NAME OpenCV
    VERSION 3.4.13
    URL https://download.fastgit.org/yjmm10/Public/releases/download/opencv/opencv-3.4.13-staticLib.7z
    OPTIONS    DOWNLOAD_ONLY
)
message("OPENCV 路径：${OpenCV_SOURCE_DIR}")
include(${OpenCV_SOURCE_DIR}/windows-x86/OpenCVConfig.cmake)
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif (OpenCV_FOUND)

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "OpenMP Not Found!")
endif ()

# OnnxRuntime
CPMaddPackage(NAME OnnxRuntime
    VERSION 1.9.0
    URL https://download.fastgit.org/RapidAI/OnnxruntimeBuilder/releases/download/1.9.0/onnxruntime-1.9.0-vs2019-shared.7z
    OPTIONS     DOWNLOAD_ONLY
)
include(${OnnxRuntime_SOURCE_DIR}/windows-x86/OnnxRuntimeConfig.cmake)
find_package(OnnxRuntime REQUIRED)
if (OnnxRuntime_FOUND)
    message(STATUS "OnnxRuntime_LIBS: ${OnnxRuntime_LIBS}")
    message(STATUS "OnnxRuntime_INCLUDE_DIRS: ${OnnxRuntime_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "onnxruntime Not Found!")
endif (OnnxRuntime_FOUND)

add_subdirectory(utils)
add_subdirectory(DbNet)
add_subdirectory(AngleNet)
add_subdirectory(CrnnNet)
add_subdirectory(OcrLite)
add_subdirectory(EXE)

install(DIRECTORY 
            ${OnnxRuntime_SOURCE_DIR}/windows-x86/bin/ 
        DESTINATION bin 
            PATTERN ".dll")
