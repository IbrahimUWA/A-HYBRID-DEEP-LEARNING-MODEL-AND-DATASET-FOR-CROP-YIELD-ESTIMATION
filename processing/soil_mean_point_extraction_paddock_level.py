import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import os
# Function to extract statistics (mean, etc.) from raster within the boundary

# The script calculates the mean value of raster data (e.g., soil or environmental features) within each geometry (paddock boundaries)
# in a GeoDataFrame. The results are added to the GeoDataFrame as new columns, where each column corresponds 
# to a raster file's name (feature). The updated GeoDataFrame is then saved to a new GeoPackage file.

def extract_raster_stats(gdf, raster_path, column_prefix):
    # Open the raster file using rasterio
    with rasterio.open(raster_path) as src:
        for index, row in gdf.iterrows():
            # Get the geometry in the same CRS as the raster
            geom = [row['geometry']]
            try:
                out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                out_image = out_image[0]  # Extract the first band if multiple
                # Filter out nodata values
                valid_pixels = out_image[out_image != src.nodata]

                if valid_pixels.size > 0:  # Ensure there are valid pixels to compute statistics
                    mean_value = np.nanmean(valid_pixels)
                    gdf.at[index, f"{column_prefix}_mean"] = mean_value
                else:
                    gdf.at[index, f"{column_prefix}_mean"] = np.nan
            except Exception as e:
                print(f"Error processing geometry {index}: {e}")
                gdf.at[index, f"{column_prefix}_mean"] = np.nan
                
    return gdf

# Load the GeoDataFrame containing the paddock boundaries
gdf = gpd.read_file(
    "C://Users//M.Ibrah//OneDrive - Department of Primary Industries And Regional Development//Desktop//Updated-Code-FoodAgility-Dec-24//Complete dataset//New_boundaries__Soil_data-12-2024-1.gpkg"
)

# Directory containing raster files
raster_folder = "F://Food Agility Data//Soil Features//DEM_Data//"
# List all .tif files in the raster folder
raster_files = [f for f in os.listdir(raster_folder) if f.endswith(".tif")]
# Loop over each raster file and process the data for each soil parameter
for raster_file in raster_files:
    # Derive the column prefix from the raster filename (without extension)
    column_name = os.path.splitext(raster_file)[0]
    raster_path = os.path.join(raster_folder, raster_file)
    print(f"Processing {column_name}...")
    gdf = extract_raster_stats(gdf, raster_path, column_name)

# Save the updated GeoDataFrame to a new file
output_path = "C://Users//M.Ibrah//OneDrive - Department of Primary Industries And Regional Development//Desktop//Updated-Code-FoodAgility-Dec-24//Complete dataset//New_boundaries__Soil_data-12-2024-1.gpkg"
gdf.to_file(output_path, driver="GPKG")
print(f"Updated GeoDataFrame saved to {output_path}.")