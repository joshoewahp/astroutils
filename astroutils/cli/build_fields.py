import logging
from pathlib import Path

import click

from astroutils.io import build_field_csv, build_vlass_field_csv, get_config
from astroutils.logger import setupLogger

config = get_config()
aux_path = config["DATA"]["aux_path"]

logger = logging.getLogger(__name__)


@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging mode")
@click.option(
    "-t", "--tiletype", type=click.Choice(["TILES", "COMBINED"]), default="TILES"
)
@click.argument("epoch")
def main(epoch, tiletype, verbose):
    setupLogger(verbose)

    if "vlass" in str(epoch):
        tilestr = ""
        fields = build_vlass_field_csv(epoch)
    else:
        tilestr = f"_{tiletype.lower()}"
        fields = build_field_csv(epoch, tiletype)

    fields.to_csv(f"{aux_path}/fields/{epoch}{tilestr}_fields.csv", index=False)

    logger.info(f"Created field metadata csv for {epoch}:\n{fields}")

    return


if __name__ == "__main__":
    main()
