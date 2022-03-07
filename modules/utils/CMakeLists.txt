cmake_minimum_required(VERSION 3.14)

project(utils VERSION 0.1.0)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)

add_library(${PROJECT_NAME} ${SRCS})

target_link_libraries(${PROJECT_NAME} 
    PUBLIC ${OpenCV_LIBS} )

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

file(GLOB include_file "${CMAKE_SOURCE_DIR}/*.h")
message("所有的头文件${include_file}")
set_target_properties(${PROJECT_NAME} PROPERTIES 
    PUBLIC_HEADER 
        "${include_file}")


packageProject(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    NAMESPACE ${PROJECT_NAME}
    BINARY_DIR ${PROJECT_BINARY_DIR}
    INCLUDE_DIR ${PROJECT_SOURCE_DIR}
    INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
    VERSION_HEADER "${VERSION_HEADER_LOCATION}"
    COMPATIBILITY SameMajorVersion
    # DEPENDENCIES "utils 0.1.0"
    )

# install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
# ARCHIVE DESTINATION lib
# LIBRARY DESTINATION lib
# RUNTIME DESTINATION bin
# PUBLIC_HEADER DESTINATION include/${PROJECT_NAME})