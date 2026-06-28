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

double variance_from_sum(double sum, double sum_sq, std::size_t count) {
    if (count == 0) {
        return 0.0;
    }
    const double n = static_cast<double>(count);
    const double value_mean = sum / n;
    return std::max((sum_sq / n) - (value_mean * value_mean), 0.0);
}

}  // namespace

HybridHeatmaps compute_hybrid_heatmaps(
    const std::vector<double>& measured_profile,
    const std::vector<std::vector<double>>& reference_profiles,
    std::size_t usable_offsets,
    double alpha,
    double beta,
    double msd_scale_m2
) {
    if (measured_profile.empty()) {
        throw std::invalid_argument("measured_profile must not be empty");
    }
    if (msd_scale_m2 <= 0.0) {
        throw std::invalid_argument("msd_scale_m2 must be positive");
    }

    const std::size_t window_length = measured_profile.size();
    const double measured_mean = mean(measured_profile);
    double measured_var = 0.0;
    for (double value : measured_profile) {
        const double delta = value - measured_mean;
        measured_var += delta * delta;
    }
    measured_var /= static_cast<double>(window_length);
    const double measured_std = std::sqrt(std::max(measured_var, 0.0));

    HybridHeatmaps result;
    result.combined.reserve(reference_profiles.size());
    result.ncc.reserve(reference_profiles.size());
    result.msd.reserve(reference_profiles.size());

    const double weight_sum = alpha + beta;
    for (const auto& reference : reference_profiles) {
        if (reference.size() < window_length) {
            throw std::invalid_argument("reference profile is shorter than measured_profile");
        }
        const std::size_t row_offsets = std::min(usable_offsets, reference.size() - window_length + 1);
        std::vector<double> combined_row(row_offsets, 0.0);
        std::vector<double> ncc_row(row_offsets, 0.0);
        std::vector<double> msd_row(row_offsets, 0.0);

        for (std::size_t offset = 0; offset < row_offsets; ++offset) {
            double ref_sum = 0.0;
            double ref_sum_sq = 0.0;
            double numerator = 0.0;
            double squared_error_sum = 0.0;

            for (std::size_t idx = 0; idx < window_length; ++idx) {
                const double ref_value = reference[offset + idx];
                const double measured_value = measured_profile[idx];
                ref_sum += ref_value;
                ref_sum_sq += ref_value * ref_value;
                numerator += ref_value * (measured_value - measured_mean);
                const double error = ref_value - measured_value;
                squared_error_sum += error * error;
            }

            const double ref_std = std::sqrt(variance_from_sum(ref_sum, ref_sum_sq, window_length));
            const double denominator = static_cast<double>(window_length) * measured_std * ref_std;
            const double ncc_value = denominator > 0.0 ? numerator / denominator : 0.0;
            const double msd_value = 1.0 / (1.0 + ((squared_error_sum / static_cast<double>(window_length)) / msd_scale_m2));
            double combined_value = 0.0;
            if (weight_sum <= 1e-12) {
                combined_value = 0.5 * (ncc_value + msd_value);
            } else {
                combined_value = ((alpha * ncc_value) + (beta * msd_value)) / weight_sum;
            }

            ncc_row[offset] = ncc_value;
            msd_row[offset] = msd_value;
            combined_row[offset] = combined_value;
        }

        result.combined.push_back(std::move(combined_row));
        result.ncc.push_back(std::move(ncc_row));
        result.msd.push_back(std::move(msd_row));
    }

    return result;
}
