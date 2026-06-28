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
    if (array.ndim() != 1) {
        throw std::invalid_argument("measured_profile must be a 1D array");
    }
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
        "compute_hybrid_heatmaps",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& measured_profile,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& reference_profiles,
           std::size_t usable_offsets,
           double alpha,
           double beta,
           double msd_scale_m2) {
            const auto result = compute_hybrid_heatmaps(
                to_vector_1d(measured_profile),
                to_vector_2d(reference_profiles),
                usable_offsets,
                alpha,
                beta,
                msd_scale_m2
            );
            return py::dict(
                "combined"_a = result.combined,
                "ncc"_a = result.ncc,
                "msd"_a = result.msd
            );
        },
        py::arg("measured_profile"),
        py::arg("reference_profiles"),
        py::arg("usable_offsets"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("msd_scale_m2")
    );
}
