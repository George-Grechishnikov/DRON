#include "correlation_core.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

std::vector<double> to_vector_1d(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) {
    std::vector<double> result(static_cast<std::size_t>(array.size()));
    std::memcpy(result.data(), array.data(), static_cast<std::size_t>(array.size()) * sizeof(double));
    return result;
}

std::vector<std::vector<double>> to_vector_2d(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array
) {
    if (array.ndim() != 2) {
        throw std::invalid_argument("reference_profiles must be a 2D array");
    }
    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(array.shape(0)));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        std::vector<double> values(static_cast<std::size_t>(array.shape(1)));
        for (py::ssize_t col = 0; col < array.shape(1); ++col) {
            values[static_cast<std::size_t>(col)] = *array.data(row, col);
        }
        result.push_back(std::move(values));
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_terrain_nav_core, module) {
    module.def(
        "normalized_correlation",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& measured_profile,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& reference_profile) {
            return normalized_correlation(to_vector_1d(measured_profile), to_vector_1d(reference_profile));
        }
    );
    module.def(
        "correlate_profiles",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& measured_profile,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& reference_profiles,
           std::ptrdiff_t max_offset_steps) {
            return correlate_profiles(to_vector_1d(measured_profile), to_vector_2d(reference_profiles), max_offset_steps);
        },
        py::arg("measured_profile"),
        py::arg("reference_profiles"),
        py::arg("max_offset_steps") = -1
    );
    module.def(
        "find_best_match",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& measured_profile,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& reference_profiles,
           std::ptrdiff_t max_offset_steps) {
            auto [best_azimuth_idx, best_offset_idx, best_score, heatmap] =
                find_best_match(to_vector_1d(measured_profile), to_vector_2d(reference_profiles), max_offset_steps);
            return py::dict(
                "best_azimuth_idx"_a = best_azimuth_idx,
                "best_offset_idx"_a = best_offset_idx,
                "best_score"_a = best_score,
                "heatmap"_a = heatmap
            );
        },
        py::arg("measured_profile"),
        py::arg("reference_profiles"),
        py::arg("max_offset_steps") = -1
    );
}
