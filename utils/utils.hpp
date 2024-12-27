#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <chrono>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <vector>

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
