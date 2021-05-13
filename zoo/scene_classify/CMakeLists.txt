cmake_minimum_required(VERSION 3.1.0)

if(${CMAKE_VERSION} VERSION_LESS 3.11)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.11)
endif()

project(DeepvacCLS)

set(CMAKE_BUILD_TYPE "RELEASE")

file(GLOB CLS_SRC src/*.cpp)
file(GLOB CLS_HEADERS include/*.h)

add_library(DeepvacCLS STATIC ${CLS_SRC})

target_include_directories(DeepvacCLS PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../..>
)

