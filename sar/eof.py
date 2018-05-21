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

import requests

import bs4
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

BASE_URL = "https://qc.sentinel1.eo.esa.int/aux_poeorb/"


def download_eof_list(start_date=None):
    if isinstance(start_date, str):
        start_date = parse(start_date)

    url = BASE_URL + '?validity_start_time={}'.format(start_date.strftime('%Y-%m-%d'))
    response = requests.get(url)
    response.raise_for_status()

    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=lambda link: link.endswith('.EOF'))

    return [link.text for link in links]


def download_eof(orbit_date, sentinel_mission=None):
    if isinstance(orbit_date, str):
        orbit_date = parse(orbit_date)

    # Subtract one day from desired orbit to get correct file
    validity_date = orbit_date + relativedelta(days=-1)

    eof_links = download_eof_list(validity_date)
    if sentinel_mission:
        eof_links = [link for link in eof_links if sentinel_mission in link]

    for link in eof_links:
        print('Downloading {}'.format(link))
        response = requests.get(BASE_URL + link)
        response.raise_for_status()
        with open(link, 'wb') as f:
            print('Saving {} to file'.format(link))
            f.write(response.content)
