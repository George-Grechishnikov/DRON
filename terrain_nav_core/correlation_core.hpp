#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

struct HybridHeatmaps {
    std::vector<std::vector<double>> combined;
    std::vector<std::vector<double>> ncc;
    std::vector<std::vector<double>> msd;
};

HybridHeatmaps compute_hybrid_heatmaps(
    const std::vector<double>& measured_profile,
    const std::vector<std::vector<double>>& reference_profiles,
    std::size_t usable_offsets,
    double alpha,
    double beta,
    double msd_scale_m2
);
