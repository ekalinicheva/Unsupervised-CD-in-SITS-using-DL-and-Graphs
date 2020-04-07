from osgeo import gdal, gdal_array, ogr, osr
import numpy as np


driver_tiff = gdal.GetDriverByName("GTiff")
driver_shp = ogr.GetDriverByName("ESRI Shapefile")


# Create GeoTIFF
def create_tiff(nb_channels, new_tiff_name, width, height, datatype, data_array, geotransformation, projection):
    dst_ds = driver_tiff.Create(new_tiff_name, width, height, nb_channels, datatype)
    if nb_channels == 1:
        dst_ds.GetRasterBand(1).WriteArray(data_array)
    else:
        for ch in range(nb_channels):
            dst_ds.GetRasterBand(ch + 1).WriteArray(data_array[ch])
    dst_ds.SetGeoTransform(geotransformation)
    dst_ds.SetProjection(projection)
    return dst_ds


# Vectorize GeoTIFF dataset
def vectorize_tiff(main_path, shp_name, ds_tiff):
    band = ds_tiff.GetRasterBand(1)
    dst_ds_shp = driver_shp.CreateDataSource((main_path + shp_name + ".shp"))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds_tiff.GetProjectionRef())
    dst_layer = dst_ds_shp.CreateLayer(shp_name, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('value', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    gdal.Polygonize(band, None, dst_layer, 0, [], callback=None)
    dst_ds_shp.Destroy()
    band = None
    ds_tiff = None
    dst_ds_shp = None


# Open GeoTIFF as an array
def open_tiff(path, name):
    ds = gdal.Open(path+"/"+name+".TIF")
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    bands_nb = ds.RasterCount
    W = ds.RasterXSize
    H = ds.RasterYSize
    try:
        image_array = gdal_array.LoadFile(path + "/" + name+".TIF")
    except:
        image_array = gdal_array.LoadFile(path + name+".TIF")
    ds = None
    return np.asarray(image_array), H, W, geo, proj, bands_nb
