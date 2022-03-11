cmake_minimum_required(VERSION 3.14)

project(core VERSION 0.1.0)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)

add_library(${PROJECT_NAME} ${SRCS})

target_link_libraries(${PROJECT_NAME} 
    PUBLIC ${OpenCV_LIBS} ${OnnxRuntime_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)


file(GLOB include_file ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

set(public_include  ${CMAKE_CURRENT_SOURCE_DIR}/types.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/operators.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/inference.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/modernocr.h)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${public_include}")
    # "${CMAKE_CURRENT_SOURCE_DIR}/types.h;
    #  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h;
    #  ${CMAKE_CURRENT_SOURCE_DIR}/operators.h;
    #  ${CMAKE_CURRENT_SOURCE_DIR}/ort.h")

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
ARCHIVE DESTINATION lib
LIBRARY DESTINATION lib
RUNTIME DESTINATION bin
PUBLIC_HEADER DESTINATION include/${PROJECT_NAME})
