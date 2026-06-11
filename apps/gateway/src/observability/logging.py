import logging
import os

from pythonjsonlogger import jsonlogger


TRACE_LEVEL = 5


def _install_trace_level() -> None:
    if logging.getLevelName(TRACE_LEVEL) != "TRACE":
        logging.addLevelName(TRACE_LEVEL, "TRACE")

    if not hasattr(logging.Logger, "trace"):
        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(TRACE_LEVEL):
                self._log(TRACE_LEVEL, message, args, **kwargs)

        logging.Logger.trace = trace  # type: ignore[attr-defined]


def configure_logging() -> None:
    _install_trace_level()
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = TRACE_LEVEL if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger(__name__).debug(
        "Logging configured",
        extra={"configured_log_level": level_name, "effective_log_level": logging.getLevelName(level)},
    )
