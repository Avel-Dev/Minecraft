#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector<char const *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct RendererContext
{
    // =====================================================
    // INSTANCE LEVEL
    // =====================================================

    vk::raii::Instance instance = nullptr;
    vk::raii::Context context;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    std::vector<const char *> enabledLayers;
    std::vector<const char *> enabledExtensions;

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    // =====================================================
    // LOGICAL DEVICE
    // =====================================================

    struct Device
    {
        vk::raii::Device handle;

        // Queues
        vk::raii::Queue graphicsQueue;
        vk::raii::Queue computeQueue;
        vk::raii::Queue transferQueue;
        vk::raii::Queue presentQueue;

        uint32_t graphicsFamily{};
        uint32_t computeFamily{};
        uint32_t transferFamily{};
        uint32_t presentFamily{};
    } device;

    // =====================================================
    // COMMAND INFRASTRUCTURE (GLOBAL)
    // =====================================================

    vk::raii::CommandPool graphicsCommandPool;
    vk::raii::CommandPool transferCommandPool;

    // upload/immediate command helpers
    vk::raii::Fence immediateFence;
    vk::raii::CommandBuffer immediateCmd;

    // =====================================================
    // SYNCHRONIZATION (GLOBAL ONLY)
    // =====================================================

    vk::raii::Semaphore timelineSemaphore;
    uint64_t timelineValue{};

    // =====================================================
    // DESCRIPTOR INFRASTRUCTURE
    // =====================================================

    vk::raii::DescriptorPool globalDescriptorPool;
    vk::raii::DescriptorSetLayout bindlessLayout; // optional

    // =====================================================
    // PIPELINE INFRASTRUCTURE
    // =====================================================

    VkPipelineCache pipelineCache{};

    // =====================================================
    // FORMAT + CAPABILITY CACHE
    // =====================================================

    struct Capabilities
    {
        bool timelineSemaphore{};
        bool dynamicRendering{};
        bool descriptorIndexing{};
        bool bufferDeviceAddress{};
        bool synchronization2{};
        bool rayTracing{};
    } caps;

    // commonly queried formats
    vk::Format depthFormat{};
    vk::SampleCountFlagBits msaaSamples{VK_SAMPLE_COUNT_1_BIT};

    // =====================================================
    // DEBUG UTILITIES
    // =====================================================
};

class Renderer
{
  public:
    void init(GLFWwindow *window);
    void drawFrame();
    void cleanup();

  private:
    void createInstance()
    {
        vk::ApplicationInfo appInfo{};
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = vk::ApiVersion14;

        std::vector<char const *> requiredLayers;
        if (enableValidationLayers)
        {
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        auto layerProperties = m_RendererCTX.context.enumerateInstanceLayerProperties();
        for (auto const &requiredLayer : requiredLayers)
        {
            if (std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty)
                                     { return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
            {
                throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
            }
        }

        auto requiredExtensions = getRequiredExtensions();

        auto extensionProperties = m_RendererCTX.context.enumerateInstanceExtensionProperties();
        for (auto const &requiredExtension : requiredExtensions)
        {
            if (std::ranges::none_of(extensionProperties, [requiredExtension](auto const &extensionProperty)
                                     { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
            {
                throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
            }
        }

        vk::InstanceCreateInfo createInfo{};
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
        createInfo.ppEnabledLayerNames = requiredLayers.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
        createInfo.ppEnabledExtensionNames = requiredExtensions.data();

        m_RendererCTX.instance = vk::raii::Instance(m_RendererCTX.context, createInfo);
    }
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createRenderPass();
    void createPipeline();
    void createFramebuffers();
    void createCommandBuffers();
    void createSyncObjects();

    std::vector<const char *> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers)
        {
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
        }

        return extensions;
    }

  private:
    RendererContext m_RendererCTX;
};
