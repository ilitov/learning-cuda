#include <algorithm>
#include <ranges>
#include <utils.hpp>

#include "kernel.hpp"

void sequentialVecAdd(const std::vector<float> &veca, const std::vector<float> &vecb, std::vector<float> &vecc) {
    for (auto [a, b, c] : std::views::zip(veca, vecb, vecc)) {
        c = a + b;
    }
}

int main() {
    const std::size_t N = 100'000;

    std::vector<float> a = utils::randVec<float>(N);
    std::vector<float> b = utils::randVec<float>(N);
    std::vector<float> c(a.size());

    {
        utils::Time<std::chrono::microseconds> _{"Seq"};
        sequentialVecAdd(a, b, c);
    }

    utils::DeviceVector<float> deva{a};
    utils::DeviceVector<float> devb{b};
    utils::DeviceVector<float> devc{deva.size()};

    {
        utils::Time<std::chrono::microseconds> _{"Par"};
        utils::checkCuda(parallelAdd(deva.data(), devb.data(), devc.data(), devc.size()));
    }

    std::vector<float> hostc(devc.size());
    devc.toHost(hostc);

    for (auto [a, b] : std::views::zip(c, hostc)) {
        assert(a == b);
    }

    return 0;
}
