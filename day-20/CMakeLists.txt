cmake_minimum_required(VERSION 3.16)
project(vo_frontend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# OpenCV
find_package(OpenCV 4.0 REQUIRED)

add_executable(vo_frontend vo_frontend.cpp)
target_link_libraries(vo_frontend PRIVATE ${OpenCV_LIBS})
target_include_directories(vo_frontend PRIVATE ${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build: mkdir build && cd build && cmake .. && make -j4")
