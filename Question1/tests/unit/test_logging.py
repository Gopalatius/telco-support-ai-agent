from telco_agent.infrastructure.logging import configure_logging


def test_configure_logging_sets_structlog_and_stdlib(monkeypatch) -> None:
    configure_calls: list[dict[str, object]] = []
    basic_config_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "telco_agent.infrastructure.logging.structlog.configure",
        lambda **kwargs: configure_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "telco_agent.infrastructure.logging.logging.basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )

    configure_logging()

    assert len(configure_calls) == 1
    assert configure_calls[0]["cache_logger_on_first_use"] is True
    assert len(basic_config_calls) == 1
    assert basic_config_calls[0]["level"] is not None
