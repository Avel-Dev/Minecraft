#include <algorithm>
#include <array>
#include <assert.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <stdint.h>
#include <thread>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint32_t PARTICLE_COUNT = 8192;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

struct UniformBufferObject
{
    float deltaTime = 1.0f;
};

struct Particle
{
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return {0, sizeof(Particle), vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color)),
        };
    }
};

// Simple logging function
template <typename... Args> void log(Args &&...args)
{
    // Only log in debug builds
#ifdef _DEBUG
    (std::cout << ... << std::forward<Args>(args)) << std::endl;
#endif
}

class ThreadSafeResourceManager
{
  private:
    std::mutex resourceMutex;
    std::vector<vk::raii::CommandPool> commandPools;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

  public:
    void createThreadCommandPools(vk::raii::Device &device, uint32_t queueFamilyIndex, uint32_t threadCount)
    {
        std::lock_guard<std::mutex> lock(resourceMutex);

        commandBuffers.clear();
        commandPools.clear();

        for (uint32_t i = 0; i < threadCount; i++)
        {
            vk::CommandPoolCreateInfo poolInfo{};
            poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer).setQueueFamilyIndex(queueFamilyIndex);
            try
            {
                commandPools.emplace_back(device, poolInfo);
            }
            catch (const std::exception &)
            {
                throw; // Re-throw the exception to be caught by the caller
            }
        }
    }

    vk::raii::CommandPool &getCommandPool(uint32_t threadIndex)
    {
        std::lock_guard lock(resourceMutex);
        return commandPools[threadIndex];
    }

    void allocateCommandBuffers(vk::raii::Device &device, uint32_t threadCount, uint32_t buffersPerThread)
    {
        std::lock_guard lock(resourceMutex);

        commandBuffers.clear();

        if (commandPools.size() < threadCount)
        {
            throw std::runtime_error("Not enough command pools for thread count");
        }

        for (uint32_t i = 0; i < threadCount; i++)
        {
            vk::CommandBufferAllocateInfo allocInfo{};
            allocInfo.setCommandPool(*commandPools[i])
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(buffersPerThread);
            try
            {
                auto threadBuffers = device.allocateCommandBuffers(allocInfo);
                for (auto &buffer : threadBuffers)
                {
                    commandBuffers.emplace_back(std::move(buffer));
                }
            }
            catch (const std::exception &)
            {
                throw; // Re-throw the exception to be caught by the caller
            }
        }
    }

    vk::raii::CommandBuffer &getCommandBuffer(uint32_t index)
    {
        // No need for mutex here as each thread accesses its own command buffer
        if (index >= commandBuffers.size())
        {
            throw std::runtime_error("Command buffer index out of range: " + std::to_string(index) +
                                     " (available: " + std::to_string(commandBuffers.size()) + ")");
        }
        return commandBuffers[index];
    }
};

class MultithreadedApplication
{
  public:
    void run()
    {
        initWindow();
        initVulkan();
        initThreads();
        mainLoop();
        cleanup();
    }

  private:
    GLFWwindow *window = nullptr;
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    uint32_t queueIndex = ~0;
    vk::raii::Queue queue = nullptr;
    vk::raii::SwapchainKHR swapChain = nullptr;
    vk::raii::SwapchainKHR oldSwapchain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::SurfaceFormatKHR swapChainSurfaceFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;

    std::vector<vk::raii::Buffer> shaderStorageBuffers;
    std::vector<vk::raii::DeviceMemory> shaderStorageBuffersMemory;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> graphicsCommandBuffers;

    vk::raii::Semaphore timelineSemaphore = nullptr;
    uint64_t timelineValue = 0;
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t frameIndex = 0;

    double lastFrameTime = 0.0;

    bool framebufferResized = false;

    double lastTime = 0.0f;

    uint32_t threadCount = 0;
    std::vector<std::thread> workerThreads;
    std::atomic<bool> shouldExit{false};
    std::vector<std::atomic<bool>> threadWorkReady;
    std::vector<std::atomic<bool>> threadWorkDone;

    std::mutex queueSubmitMutex;
    std::mutex workCompleteMutex;
    std::condition_variable workCompleteCv;

    ThreadSafeResourceManager resourceManager;
    struct ParticleGroup
    {
        uint32_t startIndex;
        uint32_t count;
    };
    std::vector<ParticleGroup> particleGroups;

    std::vector<const char *> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};

    // Helper functions
    [[nodiscard]] static std::vector<const char *> getRequiredInstanceExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        return extensions;
    }

    static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities)
    {
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
        {
            minImageCount = surfaceCapabilities.maxImageCount;
        }
        return minImageCount;
    }

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
    {
        assert(!availableFormats.empty());
        const auto formatIt = std::ranges::find_if(availableFormats,
                                                   [](const auto &format)
                                                   {
                                                       return format.format == vk::Format::eB8G8R8A8Srgb &&
                                                              format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
                                                   });
        return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
    {
        assert(std::ranges::any_of(availablePresentModes,
                                   [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
        return std::ranges::any_of(availablePresentModes,
                                   [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; })
                   ? vk::PresentModeKHR::eMailbox
                   : vk::PresentModeKHR::eFifo;
    }
    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) const
    {
        if (capabilities.currentExtent.width != 0xFFFFFFFF)
        {
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
    }
    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
    {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
        vk::raii::ShaderModule shaderModule{device, createInfo};

        return shaderModule;
    }
    static std::vector<char> readFile(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
        {
            throw std::runtime_error("failed to open file!");
        }
        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();

        return buffer;
    }

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Multithreading", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        lastTime = glfwGetTime();
    }

    static void framebufferResizeCallback(GLFWwindow *window, int, int)
    {
        auto app = reinterpret_cast<MultithreadedApplication *>(glfwGetWindowUserPointer(window));
        if (app)
        {
            app->framebufferResized = true;
        }
    }

    void initVulkan()
    {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createComputeDescriptorSetLayout();
        createGraphicsPipeline();
        createComputePipeline();
        createCommandPool();
        createShaderStorageBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createComputeDescriptorSets();
        createGraphicsCommandBuffers();
        createSyncObjects();
    }

    void initThreads()
    {
        // Increase thread count for better parallelism
        threadCount = 8u;
        log("Initializing ", threadCount, " threads for sequential execution");

        threadWorkReady = std::vector<std::atomic<bool>>(threadCount);
        threadWorkDone = std::vector<std::atomic<bool>>(threadCount);

        for (uint32_t i = 0; i < threadCount; i++)
        {
            threadWorkReady[i] = false;
            threadWorkDone[i] = true;
        }

        initThreadResources();

        const uint32_t particlesPerThread = PARTICLE_COUNT / threadCount;
        particleGroups.resize(threadCount);

        for (uint32_t i = 0; i < threadCount; i++)
        {
            particleGroups[i].startIndex = i * particlesPerThread;
            particleGroups[i].count =
                (i == threadCount - 1) ? (PARTICLE_COUNT - i * particlesPerThread) : particlesPerThread;
            log("Thread ", i, " will process particles ", particleGroups[i].startIndex, " to ",
                (particleGroups[i].startIndex + particleGroups[i].count - 1), " (count: ", particleGroups[i].count,
                ")");
        }

        for (uint32_t i = 0; i < threadCount; i++)
        {
            workerThreads.emplace_back(&MultithreadedApplication::workerThreadFunc, this, i);
            log("Started worker thread ", i);
        }
    }

    void workerThreadFunc(uint32_t threadIndex)
    {
        while (!shouldExit)
        {
            // Wait for work using condition variable
            {
                std::unique_lock<std::mutex> lock(workCompleteMutex);
                workCompleteCv.wait(
                    lock, [this, threadIndex]()
                    { return shouldExit || threadWorkReady[threadIndex].load(std::memory_order_acquire); });

                if (shouldExit)
                {
                    break;
                }

                if (!threadWorkReady[threadIndex].load(std::memory_order_acquire))
                {
                    continue;
                }
            }

            const ParticleGroup &group = particleGroups[threadIndex];
            bool workCompleted = false;

            try
            {
                // Get command buffer and record commands
                vk::raii::CommandBuffer *cmdBuffer = &resourceManager.getCommandBuffer(threadIndex);
                recordComputeCommandBuffer(*cmdBuffer, group.startIndex, group.count);
                workCompleted = true;
            }
            catch (const std::exception &)
            {
                workCompleted = false;
            }

            // Mark work as done
            threadWorkDone[threadIndex].store(true, std::memory_order_release);
            threadWorkReady[threadIndex].store(false, std::memory_order_release);

            // If this is not the last thread, signal the next thread to start
            if (threadIndex < threadCount - 1)
            {
                threadWorkReady[threadIndex + 1].store(true, std::memory_order_release);
            }

            // Notify main thread and other threads
            {
                std::lock_guard<std::mutex> lock(workCompleteMutex);
                workCompleteCv.notify_all();
            }
        }
    }

    void mainLoop()
    {
        const double targetFrameTime = 1.0 / 60.0;

        while (!glfwWindowShouldClose(window))
        {
            double frameStartTime = glfwGetTime();

            glfwPollEvents();
            drawFrame();

            double currentTime = glfwGetTime();
            lastFrameTime = (currentTime - lastTime) * 1000.0;
            lastTime = currentTime;

            double frameTime = currentTime - frameStartTime;

            if (frameTime < targetFrameTime)
            {
                double sleepTime = targetFrameTime - frameTime;
                std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
            }
        }

        device.waitIdle();
    }

    void cleanupSwapChain()
    {
        swapChainImageViews.clear();
        graphicsPipeline = nullptr;
        pipelineLayout = nullptr;
        computePipeline = nullptr;
        computePipelineLayout = nullptr;
        computeDescriptorSets.clear();
        computeDescriptorSetLayout = nullptr;
        descriptorPool = nullptr;

        // Unmap and clean up uniform buffers
        for (size_t i = 0; i < uniformBuffersMapped.size(); i++)
        {
            uniformBuffersMemory[i].unmapMemory();
        }
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        // Clean up shader storage buffers
        shaderStorageBuffers.clear();
        shaderStorageBuffersMemory.clear();

        swapChain = nullptr;
    }

    void recreateSwapChain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        device.waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createComputeDescriptorSetLayout();
        createGraphicsPipeline();
        createComputePipeline();
        createShaderStorageBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createComputeDescriptorSets();
    }

    void stopThreads()
    {
        shouldExit.store(true, std::memory_order_release);

        for (uint32_t i = 0; i < threadCount; i++)
        {
            threadWorkDone[i].store(true, std::memory_order_release);
            threadWorkReady[i].store(false, std::memory_order_release);
        }

        // Notify all threads in case they're waiting on the condition variable
        {
            std::lock_guard<std::mutex> lock(workCompleteMutex);
            workCompleteCv.notify_all();
        }

        for (auto &thread : workerThreads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }

        workerThreads.clear();
    }

    void initThreadResources()
    {
        resourceManager.createThreadCommandPools(device, queueIndex, threadCount);
        resourceManager.allocateCommandBuffers(device, threadCount, 1);
    }

    void cleanup()
    {
        stopThreads();

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createInstance()
    {
        vk::ApplicationInfo appInfo{};
        appInfo.pApplicationName = "Vulkan Multithreading";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = vk::ApiVersion14;

        auto extensions = getRequiredInstanceExtensions();
        vk::InstanceCreateInfo createInfo{};
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledLayerCount = 0;
        createInfo.ppEnabledLayerNames = nullptr;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        instance = vk::raii::Instance(context, createInfo);
    }

    void createSurface()
    {
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
        {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice)
    {
        // Check if the physicalDevice supports the Vulkan 1.3 API version
        bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;

        // Check if any of the queue families support graphics operations
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const &qfp)
                                                    { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

        // Check if all required physicalDevice extensions are available
        auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
        bool supportsAllRequiredExtensions = std::ranges::all_of(
            requiredDeviceExtension,
            [&availableDeviceExtensions](auto const &requiredDeviceExtension)
            {
                return std::ranges::any_of(
                    availableDeviceExtensions, [requiredDeviceExtension](auto const &availableDeviceExtension)
                    { return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
            });

        // Check if the physicalDevice supports the required features
        auto features =
            physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
                                                 vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
        bool supportsRequiredFeatures =
            features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
            features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

        // Return true if the physicalDevice meets all the criteria
        return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
    }

    void pickPhysicalDevice()
    {
        std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
        auto const devIter = std::ranges::find_if(physicalDevices, [&](auto const &physicalDevice)
                                                  { return isDeviceSuitable(physicalDevice); });
        if (devIter == physicalDevices.end())
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
        physicalDevice = *devIter;
    }

    void createLogicalDevice()
    {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports both graphics and present
        for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
        {
            if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
                (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
                physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
            {
                // found a queue family that supports both graphics and present
                queueIndex = qfpIndex;
                break;
            }
        }
        if (queueIndex == ~0)
        {
            throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
        }

        auto features = physicalDevice.getFeatures2();
        features.features.samplerAnisotropy = vk::True;
        vk::PhysicalDeviceVulkan13Features vulkan13Features;
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures;
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphoreFeatures;
        timelineSemaphoreFeatures.timelineSemaphore = vk::True;
        vulkan13Features.dynamicRendering = vk::True;
        vulkan13Features.synchronization2 = vk::True;
        extendedDynamicStateFeatures.extendedDynamicState = vk::True;
        extendedDynamicStateFeatures.pNext = &timelineSemaphoreFeatures;
        vulkan13Features.pNext = &extendedDynamicStateFeatures;
        features.pNext = &vulkan13Features;

        float queuePriority = 0.5f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};

        deviceQueueCreateInfo.queueFamilyIndex = queueIndex;
        deviceQueueCreateInfo.queueCount = 1;
        deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

        vk::DeviceCreateInfo deviceCreateInfo{};

        deviceCreateInfo.setPNext(&features)
            .setQueueCreateInfoCount(1)
            .setPQueueCreateInfos(&deviceQueueCreateInfo)
            .setEnabledExtensionCount(static_cast<uint32_t>(requiredDeviceExtension.size()))
            .setPpEnabledExtensionNames(requiredDeviceExtension.data());
        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        queue = vk::raii::Queue(device, queueIndex, 0);
    }

    void createSwapChain()
    {
        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);

        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        swapChainSurfaceFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(*surface));

        // Move old swapchain out (RAII-safe)
        oldSwapchain = std::move(swapChain);

        vk::SwapchainCreateInfoKHR createInfo =
            vk::SwapchainCreateInfoKHR{}
                .setSurface(*surface)
                .setMinImageCount(chooseSwapMinImageCount(surfaceCapabilities))
                .setImageFormat(swapChainSurfaceFormat.format)
                .setImageColorSpace(swapChainSurfaceFormat.colorSpace)
                .setImageExtent(swapChainExtent)
                .setImageArrayLayers(1)
                .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                .setImageSharingMode(vk::SharingMode::eExclusive)
                .setPreTransform(surfaceCapabilities.currentTransform)
                .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
                .setPresentMode(chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(*surface)))
                .setClipped(true)
                .setOldSwapchain(static_cast<vk::SwapchainKHR>(oldSwapchain)); // Create new swapchain
        swapChain = vk::raii::SwapchainKHR(device, createInfo);

        swapChainImages = swapChain.getImages();
    }

    void createImageViews()
    {
        assert(swapChainImageViews.empty());

        vk::ImageViewCreateInfo imageViewCreateInfo{};

        imageViewCreateInfo.setViewType(vk::ImageViewType::e2D)
            .setFormat(swapChainSurfaceFormat.format)
            .setComponents(vk::ComponentMapping(vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
                                                vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity))
            .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0,
                                                           1,      // baseMipLevel, levelCount
                                                           0, 1)); // baseArrayLayer, layerCount

        for (auto &image : swapChainImages)
        {
            imageViewCreateInfo.setImage(image);
            swapChainImageViews.emplace_back(device, imageViewCreateInfo);
        }
    }

    void createComputeDescriptorSetLayout()
    {
        std::array layoutBindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1,
                                           vk::ShaderStageFlagBits::eCompute)};

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};

        layoutInfo.setBindings(layoutBindings);

        computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    void createGraphicsPipeline()
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        // -------------------------------------------------
        // Shader stages
        // -------------------------------------------------
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex).setModule(*shaderModule).setPName("vertMain");

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment).setModule(*shaderModule).setPName("fragMain");

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        // -------------------------------------------------
        // Vertex input
        // -------------------------------------------------
        auto bindingDescription = Particle::getBindingDescription();
        auto attributeDescriptions = Particle::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.setVertexBindingDescriptionCount(1)
            .setPVertexBindingDescriptions(&bindingDescription)
            .setVertexAttributeDescriptionCount(static_cast<uint32_t>(attributeDescriptions.size()))
            .setPVertexAttributeDescriptions(attributeDescriptions.data());

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.setTopology(vk::PrimitiveTopology::ePointList).setPrimitiveRestartEnable(false);

        // -------------------------------------------------
        // Viewport
        // -------------------------------------------------
        vk::PipelineViewportStateCreateInfo viewportState{};
        viewportState.setViewportCount(1).setScissorCount(1);

        // -------------------------------------------------
        // Rasterizer
        // -------------------------------------------------
        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.setDepthClampEnable(false)
            .setRasterizerDiscardEnable(false)
            .setPolygonMode(vk::PolygonMode::eFill)
            .setCullMode(vk::CullModeFlagBits::eBack)
            .setFrontFace(vk::FrontFace::eCounterClockwise)
            .setDepthBiasEnable(false)
            .setLineWidth(1.0f);

        // -------------------------------------------------
        // Multisampling
        // -------------------------------------------------
        vk::PipelineMultisampleStateCreateInfo multisampling{};
        multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1).setSampleShadingEnable(false);

        // -------------------------------------------------
        // Color blending
        // -------------------------------------------------
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.setBlendEnable(true)
            .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
            .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
            .setColorBlendOp(vk::BlendOp::eAdd)
            .setSrcAlphaBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
            .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
            .setAlphaBlendOp(vk::BlendOp::eAdd)
            .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                               vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

        vk::PipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.setLogicOpEnable(false).setLogicOp(vk::LogicOp::eCopy).setAttachments(colorBlendAttachment);

        // -------------------------------------------------
        // Dynamic state
        // -------------------------------------------------
        std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

        vk::PipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.setDynamicStates(dynamicStates);

        // -------------------------------------------------
        // Pipeline layout
        // -------------------------------------------------
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        // -------------------------------------------------
        // Dynamic Rendering (StructureChain)
        // -------------------------------------------------
        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.setStages(shaderStages)
            .setPVertexInputState(&vertexInputInfo)
            .setPInputAssemblyState(&inputAssembly)
            .setPViewportState(&viewportState)
            .setPRasterizationState(&rasterizer)
            .setPMultisampleState(&multisampling)
            .setPColorBlendState(&colorBlending)
            .setPDynamicState(&dynamicState)
            .setLayout(*pipelineLayout)
            .setRenderPass(vk::RenderPass{}); // null handle

        vk::PipelineRenderingCreateInfo renderingInfo{};
        renderingInfo.setColorAttachmentFormats(swapChainSurfaceFormat.format);

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain{
            pipelineInfo, renderingInfo};

        // -------------------------------------------------
        // Create pipeline
        // -------------------------------------------------
        vk::raii::PipelineCache cache{device, {}};
        graphicsPipeline =
            vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }

    void createComputePipeline()
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));
        vk::PushConstantRange pushConstantRange = vk::PushConstantRange{}
                                                      .setStageFlags(vk::ShaderStageFlagBits::eCompute)
                                                      .setOffset(0)
                                                      .setSize(sizeof(uint32_t) * 2);

        vk::PipelineShaderStageCreateInfo computeShaderStageInfo = vk::PipelineShaderStageCreateInfo{}
                                                                       .setStage(vk::ShaderStageFlagBits::eCompute)
                                                                       .setModule(shaderModule)
                                                                       .setPName("compMain");

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo = vk::PipelineLayoutCreateInfo{}
                                                              .setSetLayouts(*computeDescriptorSetLayout)
                                                              .setPushConstantRanges(pushConstantRange);

        computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::ComputePipelineCreateInfo pipelineInfo =
            vk::ComputePipelineCreateInfo{}.setStage(computeShaderStageInfo).setLayout(*computePipelineLayout);

        computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createCommandPool()
    {
        vk::CommandPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        poolInfo.queueFamilyIndex = queueIndex;
        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    void createShaderStorageBuffers()
    {
        std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
        std::uniform_real_distribution rndDist(0.0f, 1.0f);

        std::vector<Particle> particles(PARTICLE_COUNT);
        for (auto &particle : particles)
        {
            // Generate a random position for the particle
            float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;

            // Use square root of random value to ensure uniform distribution across the area
            // This prevents clustering near the center (which causes the donut effect)
            float r = sqrtf(rndDist(rndEngine)) * 0.25f;

            float x = r * cosf(theta) * HEIGHT / WIDTH;
            float y = r * sinf(theta);
            particle.position = glm::vec2(x, y);

            // Ensure a minimum velocity and scale based on distance from center
            float minVelocity = 0.001f;
            float velocityScale = 0.003f;
            float velocityMagnitude = std::max(minVelocity, r * velocityScale);
            particle.velocity = normalize(glm::vec2(x, y)) * velocityMagnitude;
            particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
        }

        vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, particles.data(), (size_t)bufferSize);
        stagingBufferMemory.unmapMemory();

        shaderStorageBuffers.clear();
        shaderStorageBuffersMemory.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::raii::Buffer shaderStorageBufferTemp({});
            vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
            createBuffer(bufferSize,
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
                             vk::BufferUsageFlagBits::eTransferDst,
                         vk::MemoryPropertyFlagBits::eDeviceLocal, shaderStorageBufferTemp,
                         shaderStorageBufferTempMemory);
            copyBuffer(stagingBuffer, shaderStorageBufferTemp, bufferSize);
            shaderStorageBuffers.emplace_back(std::move(shaderStorageBufferTemp));
            shaderStorageBuffersMemory.emplace_back(std::move(shaderStorageBufferTempMemory));
        }
    }

    void createUniformBuffers()
    {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer,
                         bufferMem);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createDescriptorPool()
    {
        std::array poolSize{vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
                            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 2)};
        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
        poolInfo.poolSizeCount = poolSize.size();
        poolInfo.pPoolSizes = poolSize.data();
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createComputeDescriptorSets()
    {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = *descriptorPool;
        allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocInfo.pSetLayouts = layouts.data();
        computeDescriptorSets.clear();
        computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));

            vk::DescriptorBufferInfo storageBufferInfoLastFrame(
                shaderStorageBuffers[(i + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT], 0,
                sizeof(Particle) * PARTICLE_COUNT);
            vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(shaderStorageBuffers[i], 0,
                                                                   sizeof(Particle) * PARTICLE_COUNT);
            std::array<vk::WriteDescriptorSet, 3> descriptorWrites{
                vk::WriteDescriptorSet{}
                    .setDstSet(*computeDescriptorSets[i])
                    .setDstBinding(0)
                    .setDstArrayElement(0)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                    .setPBufferInfo(&bufferInfo),

                vk::WriteDescriptorSet{}
                    .setDstSet(*computeDescriptorSets[i])
                    .setDstBinding(1)
                    .setDstArrayElement(0)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setPBufferInfo(&storageBufferInfoLastFrame),

                vk::WriteDescriptorSet{}
                    .setDstSet(*computeDescriptorSets[i])
                    .setDstBinding(2)
                    .setDstArrayElement(0)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setPBufferInfo(&storageBufferInfoCurrentFrame),
            };
            device.updateDescriptorSets(descriptorWrites, {});
        }
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                      vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory) const
    {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        buffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(bufferMemory, 0);
    }

    [[nodiscard]] vk::raii::CommandBuffer beginSingleTimeCommands() const
    {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 1;
        vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        commandBuffer.begin(beginInfo);
        return commandBuffer;
    }

    void endSingleTimeCommands(const vk::raii::CommandBuffer &commandBuffer) const
    {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandBuffer;
        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    void copyBuffer(const vk::raii::Buffer &srcBuffer, const vk::raii::Buffer &dstBuffer, vk::DeviceSize size) const
    {
        vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
        commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
        endSingleTimeCommands(commandCopyBuffer);
    }

    [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const
    {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createGraphicsCommandBuffers()
    {
        graphicsCommandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        graphicsCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void recordComputeCommandBuffer(vk::raii::CommandBuffer &cmdBuffer, uint32_t startIndex, uint32_t count)
    {
        cmdBuffer.reset();

        vk::CommandBufferBeginInfo beginInfo =
            vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmdBuffer.begin(beginInfo);

        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0,
                                     {*computeDescriptorSets[frameIndex]}, {});

        struct PushConstants
        {
            uint32_t startIndex;
            uint32_t count;
        } pushConstants{startIndex, count};

        cmdBuffer.pushConstants<PushConstants>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                               pushConstants);

        uint32_t groupCount = (count + 255) / 256;
        cmdBuffer.dispatch(groupCount, 1, 1);

        cmdBuffer.end();
    }

    void recordGraphicsCommandBuffer(uint32_t imageIndex)
    {
        graphicsCommandBuffers[frameIndex].reset();

        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

        graphicsCommandBuffers[frameIndex].begin(beginInfo);

        // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
        transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eColorAttachmentOptimal,
                                {}, // srcAccessMask (no need to wait for previous operations)
                                vk::AccessFlagBits2::eColorAttachmentWrite,         // dstAccessMask
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStage
                                vk::ImageAspectFlagBits::eColor);

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        vk::RenderingAttachmentInfo attachmentInfo = vk::RenderingAttachmentInfo{}
                                                         .setImageView(swapChainImageViews[imageIndex])
                                                         .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                                                         .setLoadOp(vk::AttachmentLoadOp::eClear)
                                                         .setStoreOp(vk::AttachmentStoreOp::eStore)
                                                         .setClearValue(clearColor);

        vk::RenderingInfo renderingInfo = vk::RenderingInfo{}
                                              .setRenderArea(vk::Rect2D{}.setOffset({0, 0}).setExtent(swapChainExtent))
                                              .setLayerCount(1)
                                              .setColorAttachmentCount(1)
                                              .setPColorAttachments(&attachmentInfo);
        graphicsCommandBuffers[frameIndex].beginRendering(renderingInfo);

        graphicsCommandBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        graphicsCommandBuffers[frameIndex].setViewport(
            0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                            static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        graphicsCommandBuffers[frameIndex].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
        graphicsCommandBuffers[frameIndex].bindVertexBuffers(0, {shaderStorageBuffers[frameIndex]}, {0});
        graphicsCommandBuffers[frameIndex].draw(PARTICLE_COUNT, 1, 0, 0);
        graphicsCommandBuffers[frameIndex].endRendering();

        // After rendering, transition the swapchain image to PRESENT_SRC
        transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                                vk::ImageLayout::ePresentSrcKHR,
                                vk::AccessFlagBits2::eColorAttachmentWrite,         // srcAccessMask
                                {},                                                 // dstAccessMask
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
                                vk::PipelineStageFlagBits2::eBottomOfPipe,          // dstStage
                                vk::ImageAspectFlagBits::eColor);

        graphicsCommandBuffers[frameIndex].end();
    }

    void transition_image_layout(vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                                 vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
                                 vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask,
                                 vk::ImageAspectFlags image_aspect_flags)
    {
        vk::ImageMemoryBarrier2 barrier = vk::ImageMemoryBarrier2{}
                                              .setSrcStageMask(src_stage_mask)
                                              .setSrcAccessMask(src_access_mask)
                                              .setDstStageMask(dst_stage_mask)
                                              .setDstAccessMask(dst_access_mask)
                                              .setOldLayout(old_layout)
                                              .setNewLayout(new_layout)
                                              .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                              .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                              .setImage(image)
                                              .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                       .setAspectMask(image_aspect_flags)
                                                                       .setBaseMipLevel(0)
                                                                       .setLevelCount(1)
                                                                       .setBaseArrayLayer(0)
                                                                       .setLayerCount(1));
        vk::DependencyInfo dependency_info =
            vk::DependencyInfo{}.setDependencyFlags({}).setImageMemoryBarriers(barrier);
        graphicsCommandBuffers[frameIndex].pipelineBarrier2(dependency_info);
    }

    void signalThreadsToWork()
    {
        // Mark all threads as not done
        for (uint32_t i = 0; i < threadCount; i++)
        {
            threadWorkDone[i].store(false, std::memory_order_release);
        }

        // Memory barrier to ensure all threads see the updated threadWorkDone values
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // Only signal the first thread to start work
        threadWorkReady[0].store(true, std::memory_order_release);

        // Notify all threads in case they're waiting on the condition variable
        {
            std::lock_guard<std::mutex> lock(workCompleteMutex);
            workCompleteCv.notify_all();
        }
    }

    void waitForThreadsToComplete()
    {
        std::unique_lock<std::mutex> lock(workCompleteMutex);

        // Wait for the last thread to complete with a timeout
        auto waitResult =
            workCompleteCv.wait_for(lock, std::chrono::milliseconds(3000), [this]()
                                    { return threadWorkDone[threadCount - 1].load(std::memory_order_acquire); });

        // If we timed out, force completion
        if (!waitResult)
        {
            // Force all threads to complete
            for (uint32_t i = 0; i < threadCount; i++)
            {
                threadWorkDone[i].store(true, std::memory_order_release);
                threadWorkReady[i].store(false, std::memory_order_release);
            }

            // Notify all threads
            workCompleteCv.notify_all();
            lock.unlock();

            // Give threads a chance to respond to the forced completion
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.clear();
        inFlightFences.clear();
        vk::SemaphoreTypeCreateInfo semaphoreType =
            vk::SemaphoreTypeCreateInfo{}.setSemaphoreType(vk::SemaphoreType::eTimeline).setInitialValue(0);

        vk::SemaphoreCreateInfo createInfo = vk::SemaphoreCreateInfo{}.setPNext(&semaphoreType);

        timelineSemaphore = vk::raii::Semaphore(device, createInfo);
        timelineValue = 0;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            imageAvailableSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());

            vk::FenceCreateInfo fenceInfo;
            fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
            inFlightFences.emplace_back(device, fenceInfo);
        }
    }

    void updateUniformBuffer(uint32_t currentImage)
    {
        UniformBufferObject ubo{};
        ubo.deltaTime = static_cast<float>(lastFrameTime) * 2.0f;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame()
    {
        // Wait for the previous frame to finish
        auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
        if (fenceResult != vk::Result::eSuccess)
        {
            throw std::runtime_error("failed to wait for fence!");
        }

        // If the framebuffer was resized, rebuild the swap chain before acquiring a new image
        if (framebufferResized)
        {
            recreateSwapChain();
            framebufferResized = false;
            return;
        }

        // Acquire the next image
        auto [result, imageIndex] =
            swapChain.acquireNextImage(UINT64_MAX, *imageAvailableSemaphores[frameIndex], nullptr);

        // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined, eErrorOutOfDateKHR can be checked as
        // a result here and does not need to be caught by an exception.
        if (result == vk::Result::eErrorOutOfDateKHR)
        {
            recreateSwapChain();
            return;
        }
        // On other success codes than eSuccess and eSuboptimalKHR we just throw an exception.
        // On any error code, aquireNextImage already threw an exception.
        else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        {
            assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Update timeline values for synchronization
        uint64_t computeWaitValue = timelineValue;
        uint64_t computeSignalValue = ++timelineValue;
        uint64_t graphicsWaitValue = computeSignalValue;
        uint64_t graphicsSignalValue = ++timelineValue;

        // Update uniform buffer with the latest delta time
        updateUniformBuffer(frameIndex);

        // Signal worker threads to start processing particles
        signalThreadsToWork();

        // Record graphics command buffer while worker threads are busy
        recordGraphicsCommandBuffer(imageIndex);

        // Wait for all worker threads to complete
        waitForThreadsToComplete();

        // Collect command buffers from all threads
        std::vector<vk::CommandBuffer> computeCmdBuffers;
        computeCmdBuffers.reserve(threadCount);
        for (uint32_t i = 0; i < threadCount; i++)
        {
            try
            {
                computeCmdBuffers.push_back(*resourceManager.getCommandBuffer(i));
            }
            catch (const std::exception &)
            {
                // Skip this thread's command buffer if there was an error
            }
        }

        // Ensure we have at least one command buffer
        if (computeCmdBuffers.empty())
        {
            return;
        }

        // Set up compute submission
        vk::TimelineSemaphoreSubmitInfo computeTimelineInfo = vk::TimelineSemaphoreSubmitInfo{}
                                                                  .setWaitSemaphoreValueCount(1)
                                                                  .setPWaitSemaphoreValues(&computeWaitValue)
                                                                  .setSignalSemaphoreValueCount(1)
                                                                  .setPSignalSemaphoreValues(&computeSignalValue);
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eComputeShader};
        vk::SubmitInfo computeSubmitInfo = vk::SubmitInfo{}
                                               .setPNext(&computeTimelineInfo)
                                               .setWaitSemaphoreCount(1)
                                               .setPWaitSemaphores(&*timelineSemaphore)
                                               .setPWaitDstStageMask(waitStages)
                                               .setCommandBufferCount(static_cast<uint32_t>(computeCmdBuffers.size()))
                                               .setPCommandBuffers(computeCmdBuffers.data())
                                               .setSignalSemaphoreCount(1)
                                               .setPSignalSemaphores(&*timelineSemaphore);

        // Submit compute work
        {
            std::lock_guard<std::mutex> lock(queueSubmitMutex);
            queue.submit(computeSubmitInfo, nullptr);
        }

        // Set up graphics submission
        vk::PipelineStageFlags graphicsWaitStages[] = {vk::PipelineStageFlagBits::eVertexInput,
                                                       vk::PipelineStageFlagBits::eColorAttachmentOutput};

        std::array<vk::Semaphore, 2> waitSemaphores = {*timelineSemaphore, *imageAvailableSemaphores[frameIndex]};
        std::array<uint64_t, 2> waitSemaphoreValues = {graphicsWaitValue, 0};

        vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo =
            vk::TimelineSemaphoreSubmitInfo{}
                .setWaitSemaphoreValueCount(static_cast<uint32_t>(waitSemaphoreValues.size()))
                .setPWaitSemaphoreValues(waitSemaphoreValues.data())
                .setSignalSemaphoreValueCount(1)
                .setPSignalSemaphoreValues(&graphicsSignalValue);

        vk::SubmitInfo graphicsSubmitInfo = vk::SubmitInfo{}
                                                .setPNext(&graphicsTimelineInfo)
                                                .setWaitSemaphoreCount(static_cast<uint32_t>(waitSemaphores.size()))
                                                .setPWaitSemaphores(waitSemaphores.data())
                                                .setPWaitDstStageMask(graphicsWaitStages)
                                                .setCommandBufferCount(1)
                                                .setPCommandBuffers(&*graphicsCommandBuffers[frameIndex])
                                                .setSignalSemaphoreCount(1)
                                                .setPSignalSemaphores(&*timelineSemaphore);
        // Submit graphics work
        {
            std::lock_guard<std::mutex> lock(queueSubmitMutex);
            device.resetFences(*inFlightFences[frameIndex]);
            queue.submit(graphicsSubmitInfo, *inFlightFences[frameIndex]);
        }

        // Wait for graphics to complete before presenting
        vk::SemaphoreWaitInfo waitInfo =
            vk::SemaphoreWaitInfo{}.setSemaphores(*timelineSemaphore).setValues(graphicsSignalValue);

        auto waitResult = device.waitSemaphores(waitInfo, 5000000000);
        if (waitResult == vk::Result::eTimeout)
        {
            device.waitIdle();
            return;
        }

        // Present the image
        vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR{}
                                             .setWaitSemaphoreCount(0)
                                             .setPWaitSemaphores(nullptr)
                                             .setSwapchainCount(1)
                                             .setPSwapchains(&*swapChain)
                                             .setPImageIndices(&imageIndex);
        result = queue.presentKHR(presentInfo);
        // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined, eErrorOutOfDateKHR can be checked as
        // a result here and does not need to be caught by an exception.
        if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        else
        {
            // There are no other success codes than eSuccess; on any error code, presentKHR already threw an
            // exception.
            assert(result == vk::Result::eSuccess);
        }

        // Move to the next frame
        frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
    }
};

int main()
{
    try
    {
        MultithreadedApplication app;
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
