cmake_minimum_required(VERSION 3.16)

include(FetchContent)
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

FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	0af55ccecd98d4e5a8d1fad7de25ba429d60e863 #refs/tags/1.0.1
)

FetchContent_MakeAvailable(glm)

# Link Vulkan
target_link_libraries(VulkanApp PRIVATE glfw Vulkan::Vulkan glm::glm)
