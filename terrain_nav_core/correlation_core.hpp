#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

double normalized_correlation(
    const std::vector<double>& measured_profile,
    const std::vector<double>& reference_profile
);

std::vector<std::vector<double>> correlate_profiles(
    const std::vector<double>& measured_profile,
    const std::vector<std::vector<double>>& reference_profiles,
    std::ptrdiff_t max_offset_steps
);

std::tuple<std::size_t, std::size_t, double, std::vector<std::vector<double>>> find_best_match(
    const std::vector<double>& measured_profile,
    const std::vector<std::vector<double>>& reference_profiles,
    std::ptrdiff_t max_offset_steps
);
