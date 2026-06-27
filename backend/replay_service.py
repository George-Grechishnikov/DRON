"""Replay control adapters."""

from web_backend import controller


def start_replay(speed: float):
    return controller.start_replay(speed)


def pause_replay():
    return controller.pause_replay()


def stop_replay():
    return controller.stop_replay()


def restart_replay():
    return controller.restart_replay()


def set_speed(speed: float):
    return controller.set_speed(speed)


def force_gnss_off():
    return controller.force_gnss_off()


def force_gnss_on():
    return controller.force_gnss_on()
