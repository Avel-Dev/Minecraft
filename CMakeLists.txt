cmake_minimum_required(VERSION 3.16)

project(VulkanApp LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
# Find Vulkan
find_package(Vulkan REQUIRED)

# Create executable
add_executable(VulkanApp
    main.cpp
)

# Link Vulkan
target_link_libraries(VulkanApp PRIVATE glfw Vulkan::Vulkan)
