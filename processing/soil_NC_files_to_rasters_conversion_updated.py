import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime, timedelta

nc_file = "C:/Users/M.Ibrah/Downloads/sm_pct_variable.nc"
dataset = nc.Dataset(nc_file)

# Extract variable and time
sm_data = dataset.variables['sm_pct']
time_var = dataset.variables['time']
base_date = datetime(1900, 1, 1)
dates = np.array([base_date + timedelta(days=int(day)) for day in time_var[:]])

fill_value = -999

# Get spatial info
xmin, ymin, xmax, ymax = 111.975, -44.025, 154.025, -9.975
transform = from_bounds(xmin, ymin, xmax, ymax, 841, 681)
crs = 'EPSG:4326'

# Define years and months for extraction
years = [2020, 2021, 2022]
months = [5, 6, 7, 8, 9, 10]

for year in years:
    # Find band indices matching May-Oct of the target year
    band_indices = [i for i, date in enumerate(dates) if date.year == year and date.month in months]

    if not band_indices:
        print(f"No data found for year {year}.")
        continue

    monthly_bands = []

    with rasterio.open(f'NETCDF:"{nc_file}":sm_pct') as src:
        for band_idx in band_indices:
            band_data = src.read(band_idx + 1)  # Correct indexing

            # Replace fill_value with np.nan
            band_data = np.where(band_data == fill_value, np.nan, band_data)
            monthly_bands.append(band_data)

    monthly_bands = np.stack(monthly_bands)

    # Save individual monthly bands to raster
    with rasterio.open(
        f'C:/Users/M.Ibrah/Downloads/soil_moisture_{year}_variable.tif', 'w',
        driver='GTiff',
        height=monthly_bands.shape[1],
        width=monthly_bands.shape[2],
        count=len(band_indices),
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        for i, band in enumerate(monthly_bands, 1):
            dst.write(band, i)

    # Calculate mean ignoring NaNs
    mean_band = np.nanmean(monthly_bands, axis=0).astype(np.float32)

    # Save mean raster
    with rasterio.open(
        f'C:/Users/M.Ibrah/Downloads/sm_pct_mean_{year}_May_Oct_variable.tif',
        'w',
        driver='GTiff',
        height=mean_band.shape[0],
        width=mean_band.shape[1],
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst_mean:
        dst_mean.write(mean_band, 1)

print("Extraction and saving completed correctly.")