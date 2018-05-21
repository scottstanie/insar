#!/usr/bin/env python
"""
Utilities for handling precise orbit ephemerides (EOF) files

Example filtering URL:
?validity_start_time=2014-08&page=2


Example: 'S1A_OPER_AUX_POEORB_OPOD_20140828T122040_V20140806T225944_20140808T005944.EOF'

 'S1A' : mission id (satellite it applies to)
 'OPER' : OPER for "Routine Operations" file
 'AUX_POEORB' : AUX_ for "auxiliary data file", POEORB for Precise Orbit Ephemerides (POE) Orbit File
 'OPOD'  Site Center of the file originator

 '20140828T122040' creation date of file
 'V20140806T225944' Validity start time (when orbit is valid)
 '20140808T005944' Validity end time

Full doumentation:
https://earth.esa.int/documents/247904/349490/GMES_Sentinels_POD_Service_File_Format_Specification_GMES-GSEG-EOPG-FS-10-0075_Issue1-3.pdf

"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

import bs4
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

BASE_URL = "https://qc.sentinel1.eo.esa.int/aux_poeorb/"


def download_eofs(orbit_dates, mission=None):
    """Downloads and saves EOF files for specific dates

    Args:
        orbit_dates (list[str] or list[datetime.datetime])
        mission (str): optional to specify S1A or S1B. No input downloads both

    Returns:
        None

    Raises:
        ValueError - for mission argument not being one of 'S1A', 'S1B'
    """
    if mission and mission not in ('S1A', 'S1B'):
        raise ValueError('mission argument must be "S1A" or "S1B"')

    # Conver any string dates to a datetime
    orbit_dates = [parse(date) for date in orbit_dates if isinstance(date, str)]

    # Subtract one day from desired orbits to get correct files
    validity_dates = [date + relativedelta(days=-1) for date in orbit_dates]

    eof_links = []
    for date in validity_dates:
        cur_links = eof_link_list(date)
        if mission:
            cur_links = [link for link in cur_links if link.startswith(mission)]
        eof_links.extend(cur_links)

    # Download and save all links in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Make a dict to refer back to which link is finished downloading
        future_to_link = {executor.submit(_download_and_write, link): link for link in eof_links}
        for future in as_completed(future_to_link):
            print('Finished {}'.format(future_to_link[future]))


def eof_link_list(start_date=None):
    """
    """
    if isinstance(start_date, str):
        start_date = parse(start_date)

    url = BASE_URL + '?validity_start_time={}'.format(start_date.strftime('%Y-%m-%d'))
    response = requests.get(url)
    response.raise_for_status()

    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=lambda link: link.endswith('.EOF'))

    return [link.text for link in links]


def _download_and_write(link):
    """Wrapper function to run the link downloading in parallel

    Args:
        link (str) name of EOF file to download

    Returns:
        None
    """
    print('Downloading and saving {}'.format(link))
    response = requests.get(BASE_URL + link)
    response.raise_for_status()
    with open(link, 'wb') as f:
        f.write(response.content)
