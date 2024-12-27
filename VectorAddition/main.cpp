#include <ranges>

#include "cuda_runtime.h"
#include "utils.hpp"

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

    return 0;
}
