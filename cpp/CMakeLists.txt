cmake_minimum_required(VERSION 3.16)
project(ebhi_dip_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

add_library(ebhi_core STATIC
    src/ebhi_pipeline.cpp
    include/ebhi_pipeline.hpp
)

target_include_directories(ebhi_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ebhi_core PUBLIC
    ${OpenCV_LIBS}
    Qt6::Core
)

add_executable(ebhi_cli
    src/main_cli.cpp
)

target_link_libraries(ebhi_cli PRIVATE
    ebhi_core
)

add_executable(ebhi_gui
    src/main_gui.cpp
)

target_link_libraries(ebhi_gui PRIVATE
    ebhi_core
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)
