cmake_minimum_required(VERSION 3.14)

project(EXE VERSION 0.1.0)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
add_executable(${PROJECT_NAME} ${SRCS})

set(INTER_LIBS  core DbNet CrnnNet AngleNet OcrLite)
set(OUTER_LIBS spdlog ${OnnxRuntime_LIBS} ${OpenCV_LIBS} ${OpenMP_CXX_LIB_NAMES})
target_link_libraries(${PROJECT_NAME} 
    PUBLIC ${OUTER_LIBS} ${INTER_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
ARCHIVE DESTINATION bin
LIBRARY DESTINATION lib
RUNTIME DESTINATION bin
PUBLIC_HEADER DESTINATION include/${PROJECT_NAME})