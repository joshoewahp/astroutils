import click
import logging
from pathlib import Path

from astroutils.io import build_field_csv, build_vlass_field_csv, get_config
from astroutils.logger import setupLogger

config = get_config()
aux_path = config['DATA']['aux_path']

logger = logging.getLogger(__name__)

@click.command()
@click.option('-v', '--verbose', is_flag=True, help="Enable verbose logging mode")
@click.argument('epoch')
def main(epoch, verbose):

    setupLogger(verbose)
    
    if str(epoch) == 'vlass':
        base_dir = Path('/import/ada2/vlass/')
        fields = build_vlass_field_csv(base_dir)
    else:
        fields = build_field_csv(epoch)

    fields.to_csv(f'{aux_path}/fields/{epoch}_fields.csv', index=False)

    logger.info(f"Created field metadata csv for {epoch}:\n{fields}")
    
    return


if __name__ == '__main__':
    main()
