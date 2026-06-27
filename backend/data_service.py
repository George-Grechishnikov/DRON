"""Dataset loading and validation adapters."""

from web_backend import controller


def load_dataset(request):
    return controller.load_dataset(request)


def validate_dataset():
    return controller.validate_dataset()
