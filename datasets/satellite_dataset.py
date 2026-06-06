from torch.utils.data import Dataset
from .goes16_loader import GOES16ProxyDataset
from .insat3ds_loader import INSAT3DSProxyDataset

class SatelliteDataset(Dataset):
    """
    Factory interface routing dataset creation to GOES-16 or INSAT-3DS loaders.
    """
    def __new__(cls, satellite="GOES16", **kwargs):
        satellite_upper = satellite.upper()
        if satellite_upper == "GOES16":
            return GOES16ProxyDataset(**kwargs)
        elif satellite_upper == "INSAT3DS":
            return INSAT3DSProxyDataset(**kwargs)
        else:
            raise ValueError(
                f"Unknown satellite platform: '{satellite}'. "
                f"Supported platforms are 'GOES16' and 'INSAT3DS'."
            )
