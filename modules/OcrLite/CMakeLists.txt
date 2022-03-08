cmake_minimum_required(VERSION 3.14)

project(OcrLite VERSION 0.1.0)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)

add_library(${PROJECT_NAME} ${SRCS})

set(INTER_LIBS utils DbNet CrnnNet AngleNet)
set(OUTER_LIBS spdlog ${OnnxRuntime_LIBS} ${OpenCV_LIBS} ${OpenMP_CXX_LIB_NAMES})
target_link_libraries(${PROJECT_NAME} 
    PUBLIC ${OUTER_LIBS} ${INTER_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/OcrLite.h")

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
ARCHIVE DESTINATION lib
LIBRARY DESTINATION lib
RUNTIME DESTINATION bin
PUBLIC_HEADER DESTINATION include/${PROJECT_NAME})