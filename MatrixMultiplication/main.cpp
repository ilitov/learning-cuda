#include <algorithm>
#include <utils.hpp>

#include "kernel.hpp"

void sequentialSquareMatmul(const std::vector<float> &mata, const std::vector<float> &matb, std::vector<float> &matc, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            matc[row * n + col] = mata[row * n] * matb[col];
        }

        for (int k = 1; k < n; ++k) {
            for (int col = 0; col < n; ++col) {
                matc[row * n + col] += mata[row * n + k] * matb[row * k + col];
            }
        }
    }
}

int main() {
    const std::size_t N = 1024;

    std::vector<float> a = std::vector<float>(N * N, 1.f);
    std::vector<float> b = std::vector<float>(N * N, 1.f);
    std::vector<float> c(a.size());

    {
        utils::Time<std::chrono::milliseconds> _{"Seq"};
        sequentialSquareMatmul(a, b, c, N);
    }

    utils::DeviceVector<float> deva{a};
    utils::DeviceVector<float> devb{b};
    utils::DeviceVector<float> devc{deva.size()};

    {
        utils::Time<std::chrono::milliseconds> _{"Par"};
        parallelSquareMatmul(deva.data(), devb.data(), devc.data(), N);
    }

    std::vector<float> hostc(devc.size());
    devc.toHost(hostc);

    for (std::size_t i = 0; i < c.size(); ++i) {
        assert(c[i] == hostc[i]);
    }

    return 0;
}
