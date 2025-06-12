# ==============================================================================
# SCRIPT: NDVI and Sentinel-2 Band Median Extractor for Polygon Regions
# ------------------------------------------------------------------------------
# DESCRIPTION:
# This script extracts **median values** of Sentinel-2 Blue, Green, Red, NIR, and
# computed NDVI over polygon geometries defined in a GeoPackage (.gpkg) file.
# It processes data year-by-year using Q3 imagery from BG (Blue-Green) and NR
# (NIR-Red) band composite TIFF files.
#
# ------------------------------------------------------------------------------
# REQUIREMENTS:
# - Python 3.7+
# - Libraries: geopandas, rasterio, numpy, os
#
# Install required libraries (if needed):
#   pip install geopandas rasterio numpy
#
# ------------------------------------------------------------------------------
# EXPECTED INPUTS:
# 1. GeoPackage (`gdf_path`) with polygons and a `year` column
# 2. Folder (`raster_folder`) containing Sentinel-2 TIFF files named like:
#    - sentinel_2020_BG_Q3.tif  (Blue in Band 1, Green in Band 2)
#    - sentinel_2020_NR_Q3.tif  (NIR in Band 1, Red in Band 2)
#
# ------------------------------------------------------------------------------
# OUTPUT:
# - Returns a GeoDataFrame with the following new columns:
#     - median_blue
#     - median_green
#     - median_red
#     - median_nir
#     - median_ndvi
# - Saves a new GPKG with "_processed" appended to the original filename.
#
# ------------------------------------------------------------------------------
# USAGE:
# - Set the appropriate `gdf_path` and `raster_folder` variables.
# - Run the script:
#     python Extracting_Paddocks_Aggregation_Data_Python.py
#
# Example:
# gdf_path = "C:/path/to/your_polygons.gpkg"
# raster_folder = "F:/path/to/sentinel_data"
#
# ==============================================================================




import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import os

def process_ndvi_and_bands(gdf_path, raster_folder):
    # Read the GeoDataFrame
    gdf = gpd.read_file(gdf_path)
    
    # Sort the GeoDataFrame by year
    gdf = gdf.sort_values(by="year").reset_index(drop=True)

    # Identify the CRS from the first raster file
    first_raster = os.path.join(raster_folder, "sentinel_2020_BG_Q3.tif")
    if not os.path.exists(first_raster):
        raise FileNotFoundError("First raster file not found to determine CRS.")

    with rasterio.open(first_raster) as src:
        raster_crs = src.crs  # CRS of the raster files

    # Reproject the GeoDataFrame to the raster CRS if necessary
    if gdf.crs != raster_crs:
        print(f"Reprojecting GDF from {gdf.crs} to {raster_crs}")
        gdf = gdf.to_crs(raster_crs)

    # List to hold results
    results = []

    # Iterate over rows in the GeoDataFrame
    for _, row in gdf.iterrows():
        year = row['year']
        geometry = [row['geometry']]

        # Define raster file paths for the given year
        blue_green_file = os.path.join(raster_folder, f"sentinel_{year}_BG_Q3.tif")
        nir_red_file = os.path.join(raster_folder, f"sentinel_{year}_NR_Q3.tif")

        # Check if raster files exist
        if not os.path.exists(blue_green_file) or not os.path.exists(nir_red_file):
            print(f"Raster files for year {year} not found. Skipping...")
            continue

        try:
            # Read Blue and Green bands
            with rasterio.open(blue_green_file) as src_bg:
                out_image_bg, _ = rasterio.mask.mask(src_bg, geometry, crop=True)
                blue_band = out_image_bg[0]
                green_band = out_image_bg[1]

            # Read NIR and Red bands
            with rasterio.open(nir_red_file) as src_nr:
                out_image_nr, _ = rasterio.mask.mask(src_nr, geometry, crop=True)
                nir_band = out_image_nr[0]
                red_band = out_image_nr[1]

            # Safe NDVI Calculation to Avoid Zero Denominators
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = np.where((nir_band + red_band) == 0, 0, 
                                (nir_band.astype(float) - red_band) / (nir_band + red_band))

            # Replace NaN values with 0
            ndvi = np.nan_to_num(ndvi, nan=0)

            # Calculate median values (ignoring zero values)
            median_blue = np.median(blue_band[blue_band > 0])
            median_green = np.median(green_band[green_band > 0])
            median_red = np.median(red_band[red_band > 0])
            median_nir = np.median(nir_band[nir_band > 0])
            median_ndvi = np.median(ndvi[ndvi > 0])

            # Print NDVI values for the current row
            print(f"Year: {year}, Median NDVI: {median_ndvi:.4f}")

            # Append results to list
            results.append({
                'median_blue': median_blue,
                'median_green': median_green,
                'median_red': median_red,
                'median_nir': median_nir,
                'median_ndvi': median_ndvi
            })
        except ValueError as e:
            print(f"Error processing geometry for year {year}: {e}")
            results.append({
                'median_blue': None,
                'median_green': None,
                'median_red': None,
                'median_nir': None,
                'median_ndvi': None
            })
            continue

    # Add results back to GeoDataFrame as new columns
    gdf['median_blue'] = [res['median_blue'] for res in results]
    gdf['median_green'] = [res['median_green'] for res in results]
    gdf['median_red'] = [res['median_red'] for res in results]
    gdf['median_nir'] = [res['median_nir'] for res in results]
    gdf['median_ndvi'] = [res['median_ndvi'] for res in results]

    return gdf


# File paths
gdf_path = "C://Users//M.Ibrah//OneDrive - Department of Primary Industries And Regional Development//Desktop//Updated-Code-FoodAgility-Dec-24//Complete dataset//New_boundaries__Soil_data-12-2024-1.gpkg"
raster_folder = "F://Food Agility Data//Sentinel-2 data"

# Process and save the updated GDF
processed_gdf = process_ndvi_and_bands(gdf_path, raster_folder)
output_path = gdf_path.replace(".gpkg", "_processed.gpkg")
processed_gdf.to_file(output_path, driver="GPKG")
print(f"Processed GeoDataFrame saved to {output_path}")
