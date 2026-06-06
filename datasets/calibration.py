def dn_to_radiance(counts, slope, intercept):
    """
    Stubs converting raw digital numbers to physical radiance.
    """
    # TODO: scientific validation required
    return counts * slope + intercept

def radiance_to_brightness_temperature(radiance, band_constants):
    """
    Stubs converting thermal infrared radiance to Brightness Temperature (BT) in Kelvin.
    """
    # TODO: scientific validation required
    return radiance
