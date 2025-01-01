#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <source_location>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "cudaUtils.hpp"

namespace utils {
template <typename Duration = std::chrono::milliseconds>
struct Time {
    std::chrono::high_resolution_clock::time_point started;
    std::string_view text;
    Time(std::string_view text = "")
        : started(std::chrono::high_resolution_clock::now())
        , text(text) {
    }
    ~Time() {
        auto elapsedTime = std::chrono::high_resolution_clock::now() - started;
        std::cout << text << "; Elapsed " << std::chrono::duration_cast<Duration>(elapsedTime).count() << durationToStr()
                  << '\n';
    }
    static std::string_view durationToStr() {
        if constexpr (std::is_same_v<Duration, std::chrono::seconds>) {
            return "s";
        }
        if constexpr (std::is_same_v<Duration, std::chrono::milliseconds>) {
            return "ms";
        }
        if constexpr (std::is_same_v<Duration, std::chrono::microseconds>) {
            return "us";
        }
        if constexpr (std::is_same_v<Duration, std::chrono::nanoseconds>) {
            return "ns";
        }
    }
};

inline void checkCuda(cudaError_t error) {
    if (error != cudaError_t::cudaSuccess) {
        const auto location = std::source_location::current();
        std::cout << cudaGetErrorString(error) << " in " << location.file_name() << ':' << location.function_name() << " at "
                  << location.line() << ':' << location.column() << '\n';
        std::abort();
    }
}

template <typename T>
class DeviceVector {
    static constexpr auto CudaDelete = [](T *ptr) { checkCuda(cudaFree(ptr)); };

    using DevicePtrType = std::unique_ptr<T[], decltype(CudaDelete)>;

    static DevicePtrType allocate(std::size_t bytes) {
        if (void *devPtr{}; checkCuda(cudaMalloc(&devPtr, bytes)), devPtr) {
            return DevicePtrType{static_cast<T *>(devPtr), CudaDelete};
        }

        return {};
    }

    static void copyToDevice(T *devPtr, std::span<const T> hostData) {
        checkCuda(cudaMemcpy(devPtr, hostData.data(), hostData.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    static void copyToHost(std::span<T> hostData, const T *devPtr) {
        checkCuda(cudaMemcpy(hostData.data(), devPtr, hostData.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }

public:
    explicit DeviceVector(std::span<const T> hostData)
        : DeviceVector(hostData.size()) {
        copyToDevice(m_ptr.get(), hostData);
    }

    explicit DeviceVector(std::size_t size)
        : m_ptr(allocate(size * sizeof(T)))
        , m_size(size) {
    }

    std::size_t size() const {
        return m_size;
    }

    T *data() {
        return m_ptr.get();
    }

    const T *data() const {
        return m_ptr.get();
    }

    operator bool() const {
        return m_ptr != nullptr;
    }

    T &operator[](std::size_t index) {
        return m_ptr[index];
    }

    const T &operator[](std::size_t index) const {
        return m_ptr[index];
    }

    void toHost(std::span<T> host) const {
        assert(host.size() == size() && "Mismatch of host and device vectors!");
        copyToHost(host, m_ptr.get());
    }

private:
    DevicePtrType m_ptr;
    std::size_t m_size;
};

template <typename Container>
DeviceVector(Container) -> DeviceVector<typename Container::value_type>;

template <typename T>
inline void print(const std::vector<T> &vec) {
    for (T v : vec) {
        std::cout << v << ", ";
    }
    std::cout << '\n';
}

template <typename T>
inline std::vector<T> randVec(std::size_t n) {
    std::vector<T> vec(n);

    for (std::size_t i = 0; i < n; ++i) {
        vec[i] = static_cast<T>(i);
    }

    return vec;
}
}  // namespace utils

#endif  // !_UTILS_HPP_
