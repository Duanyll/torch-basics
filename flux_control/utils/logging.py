import logging
from rich.logging import RichHandler

def setup_rich_logging(log_level: int = logging.INFO):
    rich_handler = RichHandler()
    import transformers
    import diffusers
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.disable_progress_bar()
    transformers.utils.logging.add_handler(rich_handler)
    diffusers.utils.logging.disable_default_handler()
    diffusers.utils.logging.add_handler(rich_handler)
    diffusers.utils.logging.disable_progress_bar()
    logging.basicConfig(
        level=log_level,
        handlers=[rich_handler],
    )
