#!/usr/bin/env python
"""
Utilities for handling precise orbit ephemerides (EOF) files

Example filtering URL:
?validity_start_time=2014-08&page=2

Example EOF: 'S1A_OPER_AUX_POEORB_OPOD_20140828T122040_V20140806T225944_20140808T005944.EOF'

 'S1A' : mission id (satellite it applies to)
 'OPER' : OPER for "Routine Operations" file
 'AUX_POEORB' : AUX_ for "auxiliary data file", POEORB for Precise Orbit Ephemerides (POE) Orbit File
 'OPOD'  Site Center of the file originator

 '20140828T122040' creation date of file
 'V20140806T225944' Validity start time (when orbit is valid)
 '20140808T005944' Validity end time

Full EOF sentinel doumentation:
https://earth.esa.int/documents/247904/349490/GMES_Sentinels_POD_Service_File_Format_Specification_GMES-GSEG-EOPG-FS-10-0075_Issue1-3.pdf

API documentation: https://qc.sentinel1.eo.esa.int/doc/api/

See insar.parsers for Sentinel file naming description
"""
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    CONCURRENT = True
    MAX_WORKERS = 20
except ImportError:  # Python 2 doesn't have this :(
    CONCURRENT = False

import os
import sys
import itertools
import requests

from datetime import timedelta
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import insar.sario
from insar.log import get_log, log_runtime
from insar.parsers import Sentinel

logger = get_log()

BASE_URL = "https://qc.sentinel1.eo.esa.int/api/v1/?product_type=AUX_POEORB&validity_start__lt={start_date}&validity_stop__gt={stop_date}"
DATE_FMT = "%Y-%m-%d"  # Used in sentinel API url


def download_eofs(orbit_dates, missions=None, save_dir="."):
    """Downloads and saves EOF files for specific dates

    Args:
        orbit_dates (list[str] or list[datetime.datetime])
        missions (list[str]): optional, to specify S1A or S1B
            No input downloads both, must be same len as orbit_dates
        save_dir (str): directory to save the EOF files into

    Returns:
        None

    Raises:
        ValueError - for missions argument not being one of 'S1A', 'S1B',
            or having different length
    """
    if missions and all(m not in ('S1A', 'S1B') for m in missions):
        raise ValueError('missions argument must be "S1A" or "S1B"')
    if missions and len(missions) != len(orbit_dates):
        raise ValueError("missions arg must be same length as orbit_dates")
    if not missions:
        missions = itertools.repeat(None)

    validity_dates = list(set(orbit_dates))

    if CONCURRENT:
        # Download and save all links in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Make a dict to refer back to which date is finished downloading
            future_to_date = {
                executor.submit(_download_and_write, mission, date, save_dir): date
                for mission, date in zip(missions, validity_dates)
            }
            for future in as_completed(future_to_date):
                future.result()
                date = future_to_date[future]
                logger.info('Finished {}'.format(date))
    else:
        # Fall back for python 2:
        for mission, date in zip(missions, validity_dates):
            _download_and_write(mission, date, save_dir)
            logger.info('Finished {}'.format(date))


def eof_list(start_date):
    """Download the list of .EOF files for a specific date

    Args:
        start_date (str or datetime): Year month day of validity start for orbit file

    Returns:
        list: urls of EOF files

    Raises:
        ValueError: if start_date returns no results
    """
    if isinstance(start_date, str):
        start_date = parse(start_date)

    url = BASE_URL.format(
        start_date=start_date.strftime(DATE_FMT),
        stop_date=(start_date + timedelta(days=1)).strftime(DATE_FMT),
    )
    logger.info("Searching for EOFs at {}".format(url))
    response = requests.get(url)
    response.raise_for_status()

    if response.json()['count'] < 1:
        raise ValueError('No EOF files found for {} at {}'.format(
            start_date.strftime(DATE_FMT), url))

    return [result['remote_url'] for result in response.json()['results']]


def _download_and_write(mission, date, save_dir="."):
    """Wrapper function to run the link downloading in parallel

    Args:
        link (str) url of EOF file to download
        save_dir (str): directory to save the EOF files into

    Returns:
        None
    """
    try:
        cur_links = eof_list(date)
    except ValueError as e:  # 0 found for date
        logger.warning(e.args[0])
        logger.warning('Skipping {}'.format(date.strftime('%Y-%m-%d')))
        return

    if mission:
        cur_links = [link for link in cur_links if mission in link]

    assert len(cur_links) < 2
    for link in cur_links:
        fname = os.path.join(save_dir, link.split('/')[-1])
        if os.path.isfile(fname):
            logger.info("%s already exists, skipping download.", link)
            return

        logger.info("Downloading %s", link)
        response = requests.get(link)
        response.raise_for_status()
        logger.info("Saving to %s", fname)
        with open(fname, 'wb') as f:
            f.write(response.content)


def find_sentinel_products(startpath='./'):
    """Parse the startpath directory for any Sentinel 1 products' date and mission"""
    orbit_dates = []
    missions = []
    for filename in insar.sario.find_files(startpath, "S1*"):
        try:
            parser = Sentinel(filename)
        except ValueError:  # Doesn't match a sentinel file
            logger.info('Skipping {}'.format(filename))
            continue

        start_date = parser.start_time()
        mission = parser.mission()
        if start_date in orbit_dates:
            continue
        logger.info("Downloading precise orbits for {} on {}".format(
            mission, start_date.strftime('%Y-%m-%d')))
        orbit_dates.append(start_date)
        missions.append(mission)

    return orbit_dates, missions


@log_runtime
def main(path='.', mission=None, date=None):
    """Function used for entry point to download eofs"""
    if (mission and not date):
        logger.error("Must specify date if specifying mission.")
        sys.exit(1)
    if not date:
        # No command line args given: search current directory
        orbit_dates, missions = find_sentinel_products(path)
        if not orbit_dates:
            logger.info("No Sentinel products found in directory %s, exiting", path)
            sys.exit(0)
    if date:
        orbit_dates = [date]
        missions = list(mission) if mission else []

    download_eofs(orbit_dates, missions=missions)
