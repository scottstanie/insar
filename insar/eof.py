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

"""
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    CONCURRENT = True
except ImportError:  # Python 2 doesn't have this :(
    CONCURRENT = False

import os
import itertools
import requests

import bs4
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from insar.log import get_log

logger = get_log()

BASE_URL = "https://qc.sentinel1.eo.esa.int/aux_poeorb/"


def get_valid_dates(orbit_dates):
    """Takes a list of desired orbit dates and find the correct EOF date

    Sentinel EOFs have a validity starting one day before, and end one day after

    Args:
        list[str or datetime.datetime]: a list of dates of radar acquistions to
            get precise orbits (Sentinel 1).
            String format is flexible (uses dateutil.parser)

    Examples:
        >>> get_valid_dates(['20160102'])
        [datetime.datetime(2016, 1, 1, 0, 0)]
        >>> get_valid_dates(['01/02/2016'])
        [datetime.datetime(2016, 1, 1, 0, 0)]
        >>> import datetime
        >>> get_valid_dates(['2017-01-03', datetime.datetime(2017, 1, 5, 0, 0)])
        [datetime.datetime(2017, 1, 2, 0, 0), datetime.datetime(2017, 1, 4, 0, 0)]
    """
    # Conver any string dates to a datetime
    orbit_dates = [parse(date) if isinstance(date, str) else date for date in orbit_dates]

    # Subtract one day from desired orbits to get correct files
    return [date + relativedelta(days=-1) for date in orbit_dates]


def download_eofs(orbit_dates, missions=None):
    """Downloads and saves EOF files for specific dates

    Args:
        orbit_dates (list[str] or list[datetime.datetime])
        missions (list[str]): optional, to specify S1A or S1B
            No input downloads both, must be same len as orbit_dates

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

    validity_dates = get_valid_dates(orbit_dates)

    eof_links = []
    for mission, date in zip(missions, validity_dates):
        try:
            cur_links = eof_list(date)
        except ValueError as e:
            logger.warning(e.args[0])
            logger.warning('Skipping {}'.format(date.strftime('%Y-%m-%d')))
            continue

        if mission:
            cur_links = [link for link in cur_links if link.startswith(mission)]
        eof_links.extend(cur_links)

    if CONCURRENT:
        # Download and save all links in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Make a dict to refer back to which link is finished downloading
            future_to_link = {
                executor.submit(_download_and_write, link): link
                for link in eof_links
            }
            for future in as_completed(future_to_link):
                logger.info('Finished {}'.format(future_to_link[future]))
    else:
        # Fall back for python 2:
        for link in eof_links:
            _download_and_write(link)
            logger.info('Finished {}'.format(link))


def eof_list(start_date):
    """Download the list of .EOF files for a specific date

    Args:
        start_date (str or datetime): Year month day of validity start for orbit file

    Returns:
        list: names of EOF files

    Raises:
        ValueError: if start_date returns no results
    """
    if isinstance(start_date, str):
        start_date = parse(start_date)

    # TODO: maybe responses library for tests?
    url = BASE_URL + '?validity_start_time={}'.format(start_date.strftime('%Y-%m-%d'))
    response = requests.get(url)
    response.raise_for_status()

    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    try:
        links = soup.find_all('a', href=lambda link: link.endswith('.EOF'))
    except AttributeError:  # NoneType has no attribute .endswith
        raise ValueError('No EOF files found for {} at {}'.format(
            start_date.strftime('%Y-%m-%d'), url))

    return [link.text for link in links]


def _download_and_write(link):
    """Wrapper function to run the link downloading in parallel

    Args:
        link (str) name of EOF file to download

    Returns:
        None
    """
    if os.path.isfile(link):
        logger.info("%s already exists, skipping download.", link)
        return

    logger.info("Downloading and saving %s", link)
    response = requests.get(BASE_URL + link)
    response.raise_for_status()
    with open(link, 'wb') as f:
        f.write(response.content)
