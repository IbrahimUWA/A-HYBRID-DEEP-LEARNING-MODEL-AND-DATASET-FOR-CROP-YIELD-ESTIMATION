import os
import numpy as np
import rasterio
import rasterio.mask

def create_feature_vector(gdf, raster_folder, feature_name):
    """
    Computes the mean value of all valid pixels within each paddock and assigns it to the GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing paddock geometries and year attributes.
        raster_folder (str): Path to the folder containing raster files.
        feature_name (str): The feature name to associate the mean values.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with the computed mean values for the feature.
    """
    # Sort GeoDataFrame by year
    gdf_sorted = gdf.sort_values(by="year").reset_index(drop=True)

    for year in range(2020, 2023):  # Adjust year range as needed
        raster_filename = f'{year}.{feature_name}.tif'
        raster_path = os.path.join(raster_folder, raster_filename)
        
        # Filter paddocks for the current year
        paddocks = gdf_sorted[gdf_sorted['year'] == year]

        with rasterio.open(raster_path) as src:
            for index, row in paddocks.iterrows():
                geom = [row.geometry]  # Extract geometry as a list
                try:
                    # Mask raster data to the paddock geometry
                    out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                    out_image = out_image[0]  # Extract the first band if multiple

                    # Filter out nodata values
                    valid_pixels = out_image[out_image != src.nodata]

                    # Compute the mean value of valid pixels
                    if valid_pixels.size > 0:  # Ensure there are valid pixels to compute statistics
                        mean_value = np.nanmean(valid_pixels)
                        print(mean_value)
                    else:
                        mean_value = np.nan
                    
                    # Assign the mean value to the corresponding row in the GeoDataFrame
                    gdf_sorted.at[index, f"{feature_name}_mean"] = mean_value

                except Exception as e:
                    print(f"Error processing geometry {index}: {e}")
                    gdf_sorted.at[index, f"{feature_name}_mean"] = np.nan

    return gdf_sorted


# Example usage of the function

import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import os

# Load the GeoDataFrame containing the paddock boundaries
gdf = gpd.read_file(
    "C://Users//M.Ibrah//OneDrive - Department of Primary Industries And Regional Development//Desktop//Updated-Code-FoodAgility-Dec-24//Complete dataset//New_boundaries__Soil_data-12-2024-1.gpkg"
)

gdf_updated = create_feature_vector(gdf, 'F://Food Agility Data//Soil Features//Soil_Albedo', 'Albedo')


# Save the updated GeoDataFrame to a new file
output_path = "C://Users//M.Ibrah//OneDrive - Department of Primary Industries And Regional Development//Desktop//Updated-Code-FoodAgility-Dec-24//Complete dataset//New_boundaries__Soil_data-12-2024-1.gpkg"
gdf_updated.to_file(output_path, driver="GPKG")
print(f"Updated GeoDataFrame saved to {output_path}.")
