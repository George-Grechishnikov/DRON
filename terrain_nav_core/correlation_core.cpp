#include "correlation_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

double stddev(const std::vector<double>& values, double values_mean) {
    if (values.empty()) {
        return 0.0;
    }
    double accum = 0.0;
    for (double value : values) {
        const double delta = value - values_mean;
        accum += delta * delta;
    }
    return std::sqrt(accum / static_cast<double>(values.size()));
}

}  // namespace

double normalized_correlation(
    const std::vector<double>& measured_profile,
    const std::vector<double>& reference_profile
) {
    if (measured_profile.size() != reference_profile.size()) {
        throw std::invalid_argument("Profiles must have the same length");
    }

    const double measured_mean = mean(measured_profile);
    const double reference_mean = mean(reference_profile);
    const double measured_std = stddev(measured_profile, measured_mean);
    const double reference_std = stddev(reference_profile, reference_mean);
    if (measured_std == 0.0 || reference_std == 0.0) {
        return 0.0;
    }

    double numerator = 0.0;
    for (std::size_t idx = 0; idx < measured_profile.size(); ++idx) {
        numerator += (measured_profile[idx] - measured_mean) * (reference_profile[idx] - reference_mean);
    }
    const double denominator = static_cast<double>(measured_profile.size()) * measured_std * reference_std;
    return denominator <= 0.0 ? 0.0 : numerator / denominator;
}

std::vector<std::vector<double>> correlate_profiles(
    const std::vector<double>& measured_profile,
    const std::vector<std::vector<double>>& reference_profiles,
    std::ptrdiff_t max_offset_steps
) {
    if (measured_profile.empty()) {
        throw std::invalid_argument("measured_profile must not be empty");
    }

    std::vector<std::vector<double>> heatmap;
    for (const auto& reference_profile : reference_profiles) {
        if (reference_profile.size() < measured_profile.size()) {
            throw std::invalid_argument("reference profile is shorter than measured_profile");
        }
        const std::size_t total_offsets = reference_profile.size() - measured_profile.size() + 1;
        const std::size_t usable_offsets = max_offset_steps < 0
            ? total_offsets
            : std::min<std::size_t>(total_offsets, static_cast<std::size_t>(max_offset_steps) + 1);

        std::vector<double> row;
        row.reserve(usable_offsets);
        for (std::size_t offset = 0; offset < usable_offsets; ++offset) {
            std::vector<double> window(
                reference_profile.begin() + static_cast<std::ptrdiff_t>(offset),
                reference_profile.begin() + static_cast<std::ptrdiff_t>(offset + measured_profile.size())
            );
            row.push_back(normalized_correlation(measured_profile, window));
        }
        heatmap.push_back(std::move(row));
    }
    return heatmap;
}

std::tuple<std::size_t, std::size_t, double, std::vector<std::vector<double>>> find_best_match(
    const std::vector<double>& measured_profile,
    const std::vector<std::vector<double>>& reference_profiles,
    std::ptrdiff_t max_offset_steps
) {
    auto heatmap = correlate_profiles(measured_profile, reference_profiles, max_offset_steps);
    std::size_t best_azimuth_idx = 0;
    std::size_t best_offset_idx = 0;
    double best_score = -std::numeric_limits<double>::infinity();

    for (std::size_t azimuth_idx = 0; azimuth_idx < heatmap.size(); ++azimuth_idx) {
        for (std::size_t offset_idx = 0; offset_idx < heatmap[azimuth_idx].size(); ++offset_idx) {
            if (heatmap[azimuth_idx][offset_idx] > best_score) {
                best_score = heatmap[azimuth_idx][offset_idx];
                best_azimuth_idx = azimuth_idx;
                best_offset_idx = offset_idx;
            }
        }
    }

    return std::make_tuple(best_azimuth_idx, best_offset_idx, best_score, heatmap);
}
