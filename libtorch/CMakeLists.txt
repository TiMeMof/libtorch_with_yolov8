cmake_minimum_required(VERSION 3.15)
project(libtorchDemo)
 
# 注意此处要强制使用C++17，否则会报错
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------- libtorch --------------
find_package(PythonInterp REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "/usr/local/libtorch")
set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
message(STATUS "libTorch version: ${Torch_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")

 
add_executable(libtorchDemo run_model.cc ObjectDetector.cc)
add_executable(test test.cc)
target_link_libraries(libtorchDemo ${TORCH_LIBRARIES} ${OpenCV_LIBS} glog gflags)
target_link_libraries(test ${TORCH_LIBRARIES} ${OpenCV_LIBS})