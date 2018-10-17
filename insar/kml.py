from insar import geojson

box_template = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.2">
<GroundOverlay>
    <name> {title} </name>
    <description> {description} </description>
    <Icon>
          <href> {img_filename} </href>
    </Icon>
    <LatLonBox>
        <north> {north} </north>
        <south> {south} </south>
        <east> {east} </east>
        <west> {west} </west>
    </LatLonBox>
</GroundOverlay>
</kml>
"""

point_template = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.2">
<Placemark id="mountainpin1">
    <name>{title}</name>
    <description>{description}</description>
    <styleUrl>#pushpin</styleUrl>
    <Point>
        <coordinates>{coord_string}</coordinates>
    </Point>
</Placemark>
</kml>
"""
# This example from the Sentinel quick-look.png preview with map-overlay.kml
# Example coord_string:
# -102.2,29.5 -101.4,29.5 -101.4,28.8 -102.2,28.8 -102.2,29.5
quad_template = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:gml="http://www.opengis.net/gml" xmlns:xfdu="urn:ccsds:schema:xfdu:1" xmlns:gx="http://www.google.com/kml/ext/2.2">
<GroundOverlay>
    <name>{title}</name>
    <description>{description}</description>
    <Icon>
        <href>{img_filename}</href>
    </Icon>
    <gx:LatLonQuad>
        <coordinates>{coord_string}</coordinates>
    </gx:LatLonQuad>
</GroundOverlay>
</kml>
"""


def rsc_bounds(rsc_data):
    """Uses the x/y and step data from a .rsc file to generate LatLonBox for .kml"""
    north = rsc_data['y_first']
    west = rsc_data['x_first']
    east = west + rsc_data['width'] * rsc_data['x_step']
    south = north + rsc_data['file_length'] * rsc_data['y_step']
    return {'north': north, 'south': south, 'east': east, 'west': west}


def create_kml(rsc_data=None,
               img_filename=None,
               gj_dict=None,
               title=None,
               desc="Description",
               shape='box',
               kml_out=None,
               lon_lat=None):
    """Make a kml file to display a image (tif/png) in Google Earth

    Args:
        rsc_data (dict): dem rsc data
        img_filename (str): name of the image file
        title (str): Title for kml metadata
        desc (str): Description kml metadata
        shape (str): Options = ('box', 'quad'). Box is square, quad is arbitrary 4 sides
        kml_out (str): filename of kml to write
        lon_lat (tuple[float]): if shape == 'point', the lon and lat of the point
    """
    if title is None:
        title = img_filename

    valid_shapes = ('box', 'quad', 'point')
    if shape not in valid_shapes:
        raise ValueError("shape must be %s" % ', '.join(valid_shapes))

    if shape == 'box':
        output = box_template.format(
            title=title, description=desc, img_filename=img_filename, **rsc_bounds(rsc_data))
    elif shape == 'quad':
        output = quad_template.format(
            title=title,
            description=desc,
            img_filename=img_filename,
            coord_string=geojson.kml_string_fmt(gj_dict))
    elif shape == 'point':
        if lon_lat is None:
            # TODO: do we want to accept geojson? or overkill?
            raise ValueError("point must include lon_lat tuple")
        output = point_template.format(
            title=title, description=desc, coord_string='{},{}'.format(*lon_lat))

    if kml_out:
        print("Saving kml to %s" % kml_out)
        with open(kml_out, 'w') as f:
            f.write(output)

    return output
