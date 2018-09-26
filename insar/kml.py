def rsc_bounds(rsc_data):
    """Uses the x/y and step data from a .rsc file to generate LatLonBox for .kml"""
    north = rsc_data['y_first']
    west = rsc_data['x_first']
    east = west + rsc_data['width'] * rsc_data['x_step']
    south = north + rsc_data['file_length'] * rsc_data['y_step']
    return {'north': north, 'south': south, 'east': east, 'west': west}


def create_kml(rsc_data, img_filename, title=None, desc="Description", kml_out=None):
    """Make a kml file to display a image (tif/png) in Google Earth

    Args:
        rsc_data (dict): dem rsc data
        img_filename (str): name of the image file
        title (str): Title for kml metadata
        desc (str): Description kml metadata

        kml_out (str): filename of kml to write
    """
    if title is None:
        title = img_filename

    template = """\
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
    output = template.format(
        title=title, description=desc, img_filename=img_filename, **rsc_bounds(rsc_data))

    if kml_out:
        print("Saving kml to %s" % kml_out)
        with open(kml_out, 'w') as f:
            f.write(output)

    return output
