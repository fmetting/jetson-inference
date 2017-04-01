
file(GLOB blackjackCameraSources *.cpp)
file(GLOB blackJackCameraIncludes *.h )

cuda_add_executable(blackjack-camera ${blackjackCameraSources})
target_link_libraries(blackjack-camera nvcaffe_parser nvinfer jetson-inference)
