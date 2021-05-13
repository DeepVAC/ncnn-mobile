cmake_minimum_required(VERSION 3.1.0)

if(${CMAKE_VERSION} VERSION_LESS 3.11)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.11)
endif()

project(DeepvacAudioEmotion)

set(CMAKE_BUILD_TYPE "RELEASE")

file(GLOB AE_SRC src/*.cpp)
file(GLOB AE_HEADERS include/*.h)

add_library(DeepvacAudioEmotion STATIC ${AE_SRC})

target_include_directories(DeepvacAudioEmotion PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../..>
)
