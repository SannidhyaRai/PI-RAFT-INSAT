import os
import netCDF4 as nc
import numpy as np

def export_amv_netcdf(u_wind_array, v_wind_array, pressure_array, output_path, metadata=None):
    """
    Exports derived Atmospheric Motion Vector (AMV) velocity components and 
    Cloud-Top Pressure (CTP) vertical placement to a CF-compliant NetCDF4 file.
    
    Args:
        u_wind_array (np.ndarray): Zonal wind component (u), shape [H, W]
        v_wind_array (np.ndarray): Meridional wind component (v), shape [H, W]
        pressure_array (np.ndarray): Cloud-top pressure (P) in hPa, shape [H, W]
        output_path (str): Filepath where the NetCDF file will be written.
        metadata (dict, optional): Dict of global attributes to write.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Exporting IMD-compliant dataset container securely to: {output_path}")
    
    with nc.Dataset(output_path, 'w', format='NETCDF4') as rootgrp:
        # Establish spatial dimensions matching the array grid
        H, W = u_wind_array.shape
        rootgrp.createDimension('y', H)
        rootgrp.createDimension('x', W)
        
        # Create multi-dimensional grid variables with built-in zlib compression enabled
        u_wind = rootgrp.createVariable('u_wind', 'f4', ('y', 'x'), zlib=True)
        v_wind = rootgrp.createVariable('v_wind', 'f4', ('y', 'x'), zlib=True)
        pressure = rootgrp.createVariable('cloud_top_pressure', 'f4', ('y', 'x'), zlib=True)
        
        # Embed standard metadata attributes conforming directly to meteorological conventions
        u_wind.long_name = "Zonal Displacement Wind Velocity Component (u)"
        u_wind.units = "pixels/frame"
        u_wind.standard_name = "eastward_wind"
        
        v_wind.long_name = "Meridional Displacement Wind Velocity Component (v)"
        v_wind.units = "pixels/frame"
        v_wind.standard_name = "northward_wind"
        
        pressure.long_name = "Resolved Cloud-Top Vertical Pressure Placement (P)"
        pressure.units = "hPa"
        pressure.standard_name = "air_pressure_at_cloud_top"
        
        # Write dense numerical arrays directly into netCDF file blocks
        u_wind[:] = u_wind_array
        v_wind[:] = v_wind_array
        pressure[:] = pressure_array
        
        # Append global attributes
        if metadata:
            for key, val in metadata.items():
                setattr(rootgrp, key, val)
        else:
            rootgrp.title = "High-Resolution End-to-End Atmospheric Motion Vector (AMV) Matrix"
            rootgrp.institution = "Solutions & Engineering Team / India Meteorological Department (IMD) Collaboration"
            rootgrp.source = "INSAT-3DS Multi-Spectral Channel Processing Engine"
            rootgrp.history = "Generated via Physics-Informed Recurrent All-Pairs Field Transform (PI-RAFT) Pipeline"
            
    print(f"Successfully compiled and serialized meteorological data container to disk!")
