"""Metrics and analytics adapters."""

from web_backend import controller


def get_state():
    return controller.state()


def get_trajectory():
    return controller.trajectory()


def get_profiles():
    return controller.profiles()


def get_heatmap():
    return controller.correlation_heatmap()


def get_timeline():
    return controller.timeline()


def get_logs():
    return controller.logs_payload()


def get_metrics():
    return controller.metrics()
