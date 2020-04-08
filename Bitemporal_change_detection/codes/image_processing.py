from torch.autograd import Variable
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
    ds = gdal.Open(path+"/"+name+".TIF", 1)
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


# Extend array, used when dealing with patches, so border rows and columns are mirrored
def extend(a, patch_size):
    c=[]
    for b in range(len(a)):
        band = a[b]
        to_insert1 = np.flipud(band[1: int(patch_size / 2 + 1)])
        band = np.insert(band, 0, to_insert1, axis=0)
        to_insert2 = np.flipud(band[(len(band) - int(patch_size / 2) - 1):(len(band) - 1)])
        band = np.concatenate((band, to_insert2), axis=0)
        to_insert3 = band[:, range(int(patch_size / 2), 0, -1)]
        band = np.concatenate((to_insert3, band), axis=1)
        to_insert4 = band[:, range((len(band[0]) - 2), (len(band[0]) - int(patch_size / 2) - 2), -1)]
        band = np.concatenate((band, to_insert4), axis=1)
        c.append(list(band))
    c = np.asarray(c)
    return c


# Encode image using pretrained AE
def encode_image(encoder, loader, batch_size, nb_features, gpu, H, W, geo, proj, name_results, path_results, image_date):
    encoder.eval()
    new_coordinates = np.empty((0, nb_features), float)
    for batch_idx, (data, index) in enumerate(loader):
        if gpu:
            data = data.cuda(async=True)
            index = index.cuda(async=True)
        encoded = encoder(Variable(data))
        if gpu:
            new_coordinates_batch = encoded[0].data.cpu().numpy()
        else:
            new_coordinates_batch = encoded[0].data.numpy()
        new_coordinates = np.concatenate((new_coordinates, new_coordinates_batch), axis=0)
        if (batch_idx + 1) % 2000 == 0:
            print('Encoding : [{}/{} ({:.0f}%)]'.format(
                (batch_idx + 1) * batch_size, len(loader.dataset),
                100. * (batch_idx + 1) / len(loader)))

    # We reconstruct the image in new coordinate system
    image_array_tr = np.reshape(new_coordinates, (H, W, nb_features))
    image_array = np.transpose(image_array_tr, (2, 0, 1))
    reprojected_image_name = name_results
    reprojected_image = path_results + "Encoded_" + reprojected_image_name + ".TIF"
    dst_ds = create_tiff(nb_features, reprojected_image, W, H, gdal.GDT_Float32, image_array, geo, proj)
    dst_ds = None