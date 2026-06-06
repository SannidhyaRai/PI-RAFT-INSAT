from setuptools import setup, find_packages

setup(
    name="pi_raft",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "boto3",
        "netCDF4",
        "numpy",
        "torch",
        "matplotlib",
        "xarray",
        "pyyaml",
        "h5py",
    ],
    description="Physics-Informed Recurrent All-Pairs Field Transform (PI-RAFT) for Atmospheric Motion Vector (AMV) Retrieval",
    author="Solutions & Engineering Team / IMD Collaboration",
)
