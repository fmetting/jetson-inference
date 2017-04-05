
file(GLOB dualnetCameraSources *.cpp)
file(GLOB dualnetCameraIncludes *.h )

cuda_add_executable(dualnet-camera ${dualnetCameraSources})
target_link_libraries(dualnet-camera nvcaffe_parser nvinfer jetson-inference)
