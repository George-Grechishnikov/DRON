from checkpoint_runner import resolve_runtime_window_params


def test_resolve_runtime_window_params_clamps_to_short_upload() -> None:
    window_size, step_size = resolve_runtime_window_params(
        sample_count=15,
        requested_window_size=64,
        requested_step_size=16,
    )

    assert window_size == 15
    assert step_size == 15


def test_resolve_runtime_window_params_preserves_valid_requested_values() -> None:
    window_size, step_size = resolve_runtime_window_params(
        sample_count=200,
        requested_window_size=64,
        requested_step_size=8,
    )

    assert window_size == 64
    assert step_size == 8
