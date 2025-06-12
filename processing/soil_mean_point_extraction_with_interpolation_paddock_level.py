import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import os


from scipy.ndimage import distance_transform_edt

def extract_raster_stats_with_interpolation(gdf, raster_path, column_prefix):
    with rasterio.open(raster_path) as src:
        for index, row in gdf.iterrows():
            geom = [row['geometry']]
            try:
                # Mask the raster with the geometry
                out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                out_image = out_image[0]  # First band if multiple bands

                # Mask nodata values
                nodata = src.nodata
                valid_mask = out_image != nodata  # Mask of valid pixels

                if np.any(valid_mask):  # Check if there are valid pixels
                    # Replace NaN values with nearest neighbor interpolation
                    filled_image = replace_with_nearest(out_image, valid_mask)

                    # Compute statistics (mean)
                    mean_value = np.nanmean(filled_image)
                    gdf.at[index, f"{column_prefix}_mean"] = mean_value
                else:
                    print(f"No valid pixels for geometry {index}. Assigning NaN.")
                    gdf.at[index, f"{column_prefix}_mean"] = np.nan
            except rasterio.errors.RasterioIOError as e:
                print(f"RasterioIOError for geometry {index}. Skipping...")
                gdf.at[index, f"{column_prefix}_mean"] = np.nan
            except Exception as e:
                print(f"General error for geometry {index}: {e}")
                gdf.at[index, f"{column_prefix}_mean"] = np.nan
    return gdf

def replace_with_nearest(data, valid_mask):
    """
    Replace NaN or invalid values in the raster with the nearest valid pixel value.
    """
    # Create a distance map to the nearest valid pixel
    distances, nearest_indices = distance_transform_edt(~valid_mask, return_indices=True)

    # Replace invalid values with the nearest valid pixel value
    filled_data = data[tuple(nearest_indices)]
    filled_data[~valid_mask] = filled_data[tuple(nearest_indices)][~valid_mask]
    
    return filled_data



# Load the GeoDataFrame containing the paddock boundaries
gdf = gpd.read_file(
    "E://Updated-Rainfall-Zones-Analysis-Project//Complete dataset//New_boundaries_with_Climate_And_soil_data-12-24.gpkg"
)

# Directory containing raster files
raster_folder = "E://Food Agility Data//Soil Features//Clay"
# List all .tif files in the raster folder
raster_files = [f for f in os.listdir(raster_folder) if f.endswith(".tif")]
# Loop over each raster file and process the data for each soil parameter
for raster_file in raster_files:
    # Derive the column prefix from the raster filename (without extension)
    column_name = os.path.splitext(raster_file)[0]
    raster_path = os.path.join(raster_folder, raster_file)
    print(f"Processing {column_name}...")
    gdf = extract_raster_stats_with_interpolation(gdf, raster_path, column_name)

# Save the updated GeoDataFrame to a new file
output_path = "E://Updated-Rainfall-Zones-Analysis-Project//Complete dataset//New_boundaries_with_Climate_And_soil_data-12-2024.gpkg"
gdf.to_file(output_path, driver="GPKG")
print(f"Updated GeoDataFrame saved to {output_path}.")
