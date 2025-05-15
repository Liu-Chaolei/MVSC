# -*- coding: utf-8 -*-


import logging
import time

import click

from cogkit.cli.inference import inference
from cogkit.cli.launch import launch
from cogkit.logging import LOG_FORMAT, get_logger, set_log_level

__all__ = ["cli"]
_logger = get_logger(__name__)


@click.group()
@click.option(
    "-v",
    "--verbose",
    default=0,
    type=click.IntRange(min=0, max=2),
    count=True,
    show_default=True,
    help="Verbosity level (from 0 to 2)",
)
@click.pass_context
def cli(ctx: click.Context, verbose: int) -> None:
    start = time.perf_counter()
    logging.basicConfig(format=LOG_FORMAT, level=set_log_level(verbose))

    @ctx.call_on_close
    def _log_running_time():
        end = time.perf_counter()
        _logger.info("Running time: %.3f seconds.", end - start)


cli.add_command(inference)
cli.add_command(launch)
