from osgeo import gdal, ogr
import numpy as np
import os, re
from codes.image_processing import create_tiff, vectorize_tiff, open_tiff

driver_shp = ogr.GetDriverByName("ESRI Shapefile")  # shapefile driver


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# We obtain the BB of the segments
def get_bb(vectorGeometry, pixelWidth, pixelHeight, originX, originY):
    bb = vectorGeometry.GetEnvelope()
    x_min, x_max, y_min, y_max = bb
    x_res = abs(int((x_max - x_min) / pixelWidth))
    y_res = abs(int((y_max - y_min) / pixelHeight))
    # get x and y offset
    xOffset = int((x_min - originX) / pixelWidth)
    yOffset = int((y_max - originY) / pixelHeight)
    return x_min, x_max, y_min, y_max, x_res, y_res, xOffset, yOffset


# we create ds extent (gdal dataset datatype) of a vectorized segment
def define_extent(refLayer, refFeature, refGeometry, refSpatial, x_res, y_res, x_min, pixelWidth, y_max, pixelHeight):
    datasource = driver_shp.CreateDataSource('/vsimem/shp.shp')
    idField = ogr.FieldDefn("value", ogr.OFTInteger)
    layer = datasource.CreateLayer(" ", refSpatial, geom_type=ogr.wkbPolygon)
    layer.CreateField(idField)
    # create feature object with point geometry type from layer object:
    feature = ogr.Feature(refLayer.GetLayerDefn())
    feature.SetField('value', refFeature.GetField(0))
    feature.SetGeometry(refGeometry)
    layer.CreateFeature(feature)
    ds_clipped_extent = gdal.GetDriverByName('GTiff').Create('/vsimem/clip.tif', x_res, y_res, 1, gdal.GDT_Byte)
    ds_clipped_extent.SetGeoTransform((x_min, pixelWidth, 0, y_max, 0, pixelHeight))
    ds_clipped_extent.SetProjection(proj)
    gdal.RasterizeLayer(ds_clipped_extent, [1], layer, burn_values=[1])
    feature.Destroy()
    datasource.Destroy()
    return ds_clipped_extent


# function to open raster change images
def open_image(d1, d2, path, anomaly=False):
    if anomaly is True:
        loss_folder = \
        list(filter(lambda f: (f.startswith("Joint_AE_" + d1 + "_" + d2)), os.listdir(path+"Anomaly/")))[0] + "/"
        image_name = list(
            filter(lambda f: (f.startswith('Anomaly_Outliers_average_' + d1 + '_to_' + d2) and f.endswith(".TIF")), os.listdir(path + "Anomaly/" + loss_folder)))[0]
    else:
        loss_folder = \
        list(filter(lambda f: (f.startswith("Joint_AE_" + d1 + "_" + d2)), os.listdir(path)))[0] + "/"
        image_name = list(
            filter(lambda f: (f.startswith('Outliers_average_' + d1 + '_to_' + d2) and f.endswith(".TIF")), os.listdir(path +loss_folder)))[0]
    image_name = os.path.splitext(image_name)[0]
    image_array_outliers, H, W, geo, proj, bands_nb = open_tiff(path + loss_folder,
                                                                        image_name)
    ds = gdal.Open(path + loss_folder + image_name + ".TIF")
    return loss_folder, image_name, image_array_outliers, ds, H, W, geo, proj, bands_nb




thr_int = 0.4

# path to the folder that contains both t_t1 and t_t2 folders with corresponding change rasters for each couple of images
path_results_nn = os.path.expanduser('~/Desktop/Results/RESULTS_CHANGE_DETECTION/NN_Rostov_S2_all_images_model_pretrained/All_images_ep_1_patch_7.2019-09-04_1807_threshold_0995/')
path_results = path_results_nn + "threshold_int_" + str(thr_int) + "_17img/"
path_results_nn_t_t1 = path_results_nn + "t_t1/"
path_results_nn_t_t2 = path_results_nn + "t_t2/"


# we extract dated of change image couples
loss_folders = list(filter(lambda f: (f.startswith("Joint_AE_")), os.listdir(path_results_nn_t_t1)))
print(loss_folders)
dates = []
for folder in np.sort(loss_folders):
    date1, date2 = (re.search("Joint_AE_([0-9]*)_([0-9]*).", folder)).group(1), (re.search("Joint_AE_([0-9]*)_([0-9]*).", folder)).group(2)
    dates.append([date1, date2])
dates_couples = np.sort(np.asarray(dates))[3:]
dates = np.unique(dates_couples)
print(dates_couples)


intersection_arr_anomaly_list = []
intersection_arr_list = []
# for every couple of change images
for d in range(len(dates_couples)):
    if d != 0:
        date1, date2 = dates_couples[d]     # t, t+1
        date_prev, _ = dates_couples[d-1]   # t-1
        print(date1, date2)
        print(date_prev, date2)
        # we open change image t : t+1
        loss_folder_nn_t_t1, image_name_nn_t_t1, image_array_outliers_nn_t_t1, ds1, H, W, geo, proj, bands_nb = open_image(date1, date2, path_results_nn_t_t1)
        # we open image t-1 : t+1
        loss_folder_nn_t_t2, image_name_nn_t_t2, image_array_outliers_nn_t_t2, _, _, _, _, _, _ = open_image(date_prev, date2, path_results_nn_t_t2)
        # we open image t-1 : t
        # as we need to check the intersection with all the available changes (changes + anomalies, except FP) we sum the change raster with potential nomalies
        loss_folder_nn_t_t3, image_name_nn_t_t3, image_array_outliers_nn_t_t3, ds3, _, _, _, _, _ = open_image(date_prev, date1, path_results)
        image_array_outliers_nn_t_t3 = image_array_outliers_nn_t_t3 + intersection_arr_anomaly
        image_array_outliers_nn_t_t3[image_array_outliers_nn_t_t3 == 2] = 1


        create_dir(path_results+loss_folder_nn_t_t1)

        #intersection with t-1 : t+1
        intersection_arr = np.zeros((H, W))
        intersection_arr[(image_array_outliers_nn_t_t1 + image_array_outliers_nn_t_t2)==2]=1

        # we open segmentation results for t,t+1
        print(path_results_nn_t_t1+loss_folder_nn_t_t1+image_name_nn_t_t1)
        outliers1 = driver_shp.Open(path_results_nn_t_t1+loss_folder_nn_t_t1 + "Segments_" + image_name_nn_t_t1 + ".shp", 0)
        layer1 = outliers1.GetLayer()
        spatialRef1 = layer1.GetSpatialRef()

        # image parameters
        originX, pixelWidth, _, originY, _, pixelHeight = geo


        # we create two shapefiles. intersection will contain true changes, no_intersection will contain anomalies.
        # we will fill both layers with features in two following "for loops"
        source_intersection = driver_shp.CreateDataSource(path_results +loss_folder_nn_t_t1 + "/" + image_name_nn_t_t1 + '.shp')
        layer_intersection = source_intersection.CreateLayer(image_name_nn_t_t1, spatialRef1, geom_type=ogr.wkbPolygon)
        idField = ogr.FieldDefn("value", ogr.OFTInteger)
        layer_intersection.CreateField(idField)
        create_dir(path_results + "Anomaly/" + loss_folder_nn_t_t1)
        source_no_intersection = driver_shp.CreateDataSource(path_results + "Anomaly/" + loss_folder_nn_t_t1 + "/" + "Anomaly_" + image_name_nn_t_t1 + '.shp')
        layer_no_intersection = source_no_intersection.CreateLayer("Anomaly_" + image_name_nn_t_t1, spatialRef1, geom_type=ogr.wkbPolygon)
        layer_no_intersection.CreateField(idField)

        new_id1 = 1
        new_id2 = 1
        # we iterate through each change segment (feature1) in the shapefile
        for f1 in range(len(layer1)):
            feature1 = layer1[f1]
            id1 = feature1.GetField("value")
            if id1 > 0:
                vectorGeometry1 = feature1.GetGeometryRef()
                x_min, x_max, y_min, y_max, x_res, y_res, xOffset, yOffset = get_bb(vectorGeometry1, pixelWidth,
                                                                                    pixelHeight, originX, originY)


                ds_clipped_t_t1 = define_extent(layer1, feature1, vectorGeometry1, spatialRef1, x_res, y_res, x_min, pixelWidth, y_max,
                              pixelHeight)
                clipped_t_t1 = ds_clipped_t_t1.GetRasterBand(1).ReadAsArray()

                clipped_intersection = intersection_arr[yOffset:yOffset + y_res, xOffset:xOffset + x_res]*clipped_t_t1

                size_t_t1 = np.count_nonzero(clipped_t_t1)

                # we check if change polygon from t_t1 have sufficient intersection with polygons from t_t2
                # we also add condition that change polygon should be at least 3 pixels big
                # true changes
                if (len(np.where((clipped_intersection.flatten()+clipped_t_t1.flatten())==2)[0])/size_t_t1)>=thr_int and size_t_t1>=3:
                    feature = ogr.Feature(layer1.GetLayerDefn())
                    feature.SetField('value', new_id1)
                    feature.SetGeometry(vectorGeometry1)
                    layer_intersection.CreateFeature(feature)
                    feature.Destroy()
                    new_id1 += 1
                # potential anomalies
                elif (len(np.where((clipped_intersection.flatten()+clipped_t_t1.flatten())==2)[0])/size_t_t1)<thr_int and size_t_t1>=3:
                    feature = ogr.Feature(layer1.GetLayerDefn())
                    feature.SetField('value', new_id2)
                    feature.SetGeometry(vectorGeometry1)
                    layer_no_intersection.CreateFeature(feature)
                    feature.Destroy()
                    new_id2 += 1

        # we write raster with change polygons from t_t1 that have sufficient intersection
        ds_intersection = gdal.GetDriverByName('GTiff').Create(path_results + loss_folder_nn_t_t1 + image_name_nn_t_t1 + ".TIF", W, H, 1, gdal.GDT_Byte)
        ds_intersection.SetGeoTransform(geo)
        ds_intersection.SetProjection(proj)
        gdal.RasterizeLayer(ds_intersection, [1], layer_intersection, burn_values=[1])

        source_intersection.Destroy()
        ds_intersection = None
        intersection_arr_list.append(intersection_arr)

        # no intersection with t-1 : t+1
        # this is the raster with potential anomalies
        no_intersection_arr = image_array_outliers_nn_t_t1 - intersection_arr
        intersection_arr_anomaly = np.zeros_like(intersection_arr)
        ds_anomaly = gdal.GetDriverByName('GTiff').Create(path_results + "Anomaly/" + loss_folder_nn_t_t1 + "Anomaly_" +image_name_nn_t_t1 + ".TIF", W, H, 1, gdal.GDT_Byte)
        ds_anomaly.SetGeoTransform(geo)
        ds_anomaly.SetProjection(proj)
        gdal.RasterizeLayer(ds_anomaly, [1], layer_no_intersection, burn_values=[1])
        ds_anomaly = None


        # we purify the layer with potential anomalies and delete the anomalies that do not have intersection with n-1:n change layer
        # if they have this intersection, it means that the anomalous event happenned only once - at time t (see article), so we keep.
        # otherwise, it is just a FP change due to image imperfection
        for f2 in range(len(layer_no_intersection)):
            feature2 = layer_no_intersection[f2]
            id2 = feature2.GetField("value")
            if id2 > 0:
                vectorGeometry2 = feature2.GetGeometryRef()
                x_min, x_max, y_min, y_max, x_res, y_res, xOffset, yOffset = get_bb(vectorGeometry2, pixelWidth,
                                                                                    pixelHeight, originX, originY)

                # we open clipped anomaly
                ds_clipped_anomaly = define_extent(layer_no_intersection, feature2, vectorGeometry2, spatialRef1, x_res, y_res, x_min, pixelWidth, y_max,
                              pixelHeight)
                clipped_anomaly = ds_clipped_anomaly.GetRasterBand(1).ReadAsArray()
                clipped_t_t3 = image_array_outliers_nn_t_t3[yOffset:yOffset + y_res, xOffset:xOffset + x_res] * clipped_anomaly

                size_clipped_anomaly = np.count_nonzero(clipped_anomaly)
                if (len(np.where((clipped_anomaly.flatten()+clipped_t_t3.flatten())==2)[0])/size_clipped_anomaly)>=thr_int:
                    intersection_arr_anomaly[yOffset:yOffset + y_res, xOffset:xOffset + x_res] = clipped_anomaly
                else:
                    layer_no_intersection.DeleteFeature(feature2.GetFID())

                ds_clipped_anomaly = None



        ds_anomaly = create_tiff(1, path_results + "Anomaly/" + loss_folder_nn_t_t1 + "Anomaly_" +image_name_nn_t_t1 + ".TIF", W, H,
                                      gdal.GDT_Byte, intersection_arr_anomaly, geo,
                                      proj)
        source_no_intersection.Destroy()
        # gdal.SieveFilter(ds_anomaly.GetRasterBand(1), None, ds_anomaly.GetRasterBand(1), 3, 4)
        # ds_anomaly.FlushCache()
        # vectorize_tiff(path_results_nn + "Anomaly/" + loss_folder_nn_t_t1, "/" + "Anomaly_" +image_name_nn_t_t1, ds_anomaly)
        # intersection_arr_anomaly = ds_anomaly.GetRasterBand(1).ReadAsArray()
        intersection_arr_anomaly_list.append(intersection_arr_anomaly)
        ds_anomaly = None


    else:
        #we deal with the first CM. we cannot correct it, so we just copy it in the new folder
        date1, date2 = dates_couples[d]
        loss_folder_nn_t_t1, image_name_nn_t_t1, image_array_outliers_nn_t_t1, ds1, H, W, geo, proj, bands_nb = open_image(date1, date2, path_results_nn_t_t1)
        create_dir(path_results+loss_folder_nn_t_t1)
        create_dir(path_results + "Anomaly/" + loss_folder_nn_t_t1)
        ds_anomaly = create_tiff(1, path_results + "Anomaly/" + loss_folder_nn_t_t1 + "Anomaly_" +image_name_nn_t_t1 + ".TIF", W, H,
                                      gdal.GDT_Byte, np.zeros((H,W)), geo,
                                      proj)
        vectorize_tiff(path_results + "Anomaly/" + loss_folder_nn_t_t1, "/" + "Anomaly_" +image_name_nn_t_t1, ds_anomaly)
        ds_anomaly = None

        outliers1 = driver_shp.Open(path_results_nn_t_t1 + loss_folder_nn_t_t1 + "Segments_" + image_name_nn_t_t1 + ".shp", 0)
        layer1 = outliers1.GetLayer()
        spatialRef1 = layer1.GetSpatialRef()

        source_intersection = driver_shp.CreateDataSource(path_results +loss_folder_nn_t_t1 + "/" + image_name_nn_t_t1 + '.shp')
        layer_intersection = source_intersection.CreateLayer(image_name_nn_t_t1, spatialRef1, geom_type=ogr.wkbPolygon)
        idField = ogr.FieldDefn("value", ogr.OFTInteger)
        layer_intersection.CreateField(idField)

        new_id1=0
        for f1 in range(len(layer1)):
            feature1 = layer1[f1]
            id1 = feature1.GetField("value")
            if id1>0:
                vectorGeometry1 = feature1.GetGeometryRef()
                size_t_t1 = vectorGeometry1.GetArea()
                if size_t_t1>=300:
                    new_id1 += 1
                    feature = ogr.Feature(layer1.GetLayerDefn())
                    feature.SetField('value', new_id1)
                    feature.SetGeometry(vectorGeometry1)
                    layer_intersection.CreateFeature(feature)
                    feature.Destroy()



        ds1 = gdal.GetDriverByName('GTiff').Create(path_results + loss_folder_nn_t_t1 + image_name_nn_t_t1 + ".TIF", W, H, 1, gdal.GDT_Byte)
        ds1.SetGeoTransform(geo)
        ds1.SetProjection(proj)
        gdal.RasterizeLayer(ds1, [1], layer_intersection, burn_values=[1])
        source_intersection.Destroy()
        ds1 = None


        intersection_arr_list.append(image_array_outliers_nn_t_t1)
        intersection_arr_anomaly = np.zeros((H, W))
        intersection_arr_anomaly_list.append(intersection_arr_anomaly)


intersection_arr_list = np.asarray(intersection_arr_list)
intersection_arr_anomaly_list = np.asarray(intersection_arr_anomaly_list)


#Here we deal with potential anomalies. We check if an anomaly repeats several times in the dataset.
# If yes, it passes to normal changes, if no, it is considered as one time anomaly that does not influence the overall temporal behaviour
print("Printing anomaly percentage")
for t in range(1, len(intersection_arr_anomaly_list)):
    potential_anomaly = intersection_arr_anomaly_list[t]    #we get potential anomaly CM at date a
    new_anomaly = np.copy(potential_anomaly)
    new_previous_anomaly = np.copy(intersection_arr_anomaly_list[t-1])

    #We check CM and potential CM at other dates and define if potential anomalies at a have intersection with anomalies at other dates
    other_potential_anomalies = np.sum(np.delete(np.copy(intersection_arr_list), [t-1, t], axis=0), axis=0) + \
                                np.sum(np.delete(np.copy(intersection_arr_anomaly_list), [t], axis=0), axis=0)
    intersection_anomalies = np.zeros((H, W))
    print(t, len(np.where(potential_anomaly[other_potential_anomalies == 0] == 1)[0])/(H*W)*100)
    other_potential_anomalies[other_potential_anomalies >= 1] = 1 #this is raster with changes at other dates (1 -  change, 0 - no change)

    date1, date2 = dates_couples[t]
    date_prev, _ = dates_couples[t-1]
    print(date1, date2)
    # # we open image t : t+1
    loss_folder_nn_t_t1, image_name_nn_t_t1, image_array_outliers_nn_t_t1, _, _, _, _, _, _ = open_image(date1, date2,
                                                                                                           path_results)
    image_name_nn_anomaly = "Anomaly_" + image_name_nn_t_t1
    ds_anomaly = gdal.Open(path_results + "Anomaly/" + loss_folder_nn_t_t1 + image_name_nn_anomaly + ".TIF")
    outliers_anomaly = driver_shp.Open(path_results + "Anomaly/" + loss_folder_nn_t_t1 + image_name_nn_anomaly + ".shp", 1)
    layer_anomaly = outliers_anomaly.GetLayer()

    outliers_t1 = driver_shp.Open(path_results + loss_folder_nn_t_t1 + image_name_nn_t_t1 + ".shp", 1)
    layer1 = outliers_t1.GetLayer()


    clipped_other_dates_changes_list = []
    for ano in np.delete(np.copy(intersection_arr_list), [t-1, t], axis=0):
        clipped_other_dates_changes_list.append(ano)
    for ano in np.delete(np.copy(intersection_arr_anomaly_list), [t], axis=0):
        clipped_other_dates_changes_list.append(ano)


    #We iterate through potential anomaly polygons and check their intersection with other timestamps
    # If there is an intersection, it is not an anomaly, but a change process
    for f1 in range(len(layer_anomaly)):
        feature1 = layer_anomaly[f1]
        id1 = feature1.GetField("value")
        if id1 > 0:
            vectorGeometry1 = feature1.GetGeometryRef()
            x_min, x_max, y_min, y_max, x_res, y_res, xOffset, yOffset = get_bb(vectorGeometry1, pixelWidth, pixelHeight,
                                                                                originX, originY)

            ds_clipped_anomaly = define_extent(layer_anomaly, feature1, vectorGeometry1, spatialRef1, x_res, y_res, x_min,
                                               pixelWidth, y_max,
                                               pixelHeight)
            clipped_anomaly = ds_clipped_anomaly.GetRasterBand(1).ReadAsArray()


            clipped_potential = other_potential_anomalies[yOffset:yOffset + y_res, xOffset:xOffset + x_res]*clipped_anomaly


            size_anomaly = np.count_nonzero(clipped_anomaly)

            # we check firstly
            replace = False
            if (len(np.where((clipped_anomaly.flatten() + clipped_potential.flatten()) == 2)[0]) / size_anomaly) >= thr_int:
                for ano in clipped_other_dates_changes_list:
                    clipped_potential = ano[yOffset:yOffset + y_res, xOffset:xOffset + x_res]*clipped_anomaly
                    if (len(np.where((clipped_anomaly.flatten() + clipped_potential.flatten()) == 2)[0]) / size_anomaly) >= thr_int:
                        replace = True
            if replace:
                # it means it's not a one-time anomaly, but a change process, so we replace these pixels from anomaly folder to normal outliers
                # we put it in outliers
                image_array_outliers_nn_t_t1[yOffset:yOffset + y_res, xOffset:xOffset + x_res] = np.copy(
                    image_array_outliers_nn_t_t1[yOffset:yOffset + y_res,
                    xOffset:xOffset + x_res]) + clipped_anomaly
                feature = ogr.Feature(layer_anomaly.GetLayerDefn())
                feature.SetField('value', feature1.GetField(0))
                feature.SetGeometry(vectorGeometry1)
                layer1.CreateFeature(feature)

                # we delete it from anomaly
                new_anomaly[yOffset:yOffset + y_res, xOffset:xOffset + x_res] = np.copy(
                    new_anomaly[yOffset:yOffset + y_res, xOffset:xOffset + x_res]) - clipped_anomaly
                layer_anomaly.DeleteFeature(feature1.GetFID())
                feature.Destroy()
    new_anomaly[new_anomaly == 2] = 1
    image_array_outliers_nn_t_t1[image_array_outliers_nn_t_t1 == 2] = 1

    #We rewrite tiff with real anomalies at t:t+1
    ds_anomaly = create_tiff(1,
                             path_results + "Anomaly/" + loss_folder_nn_t_t1 + "Anomaly_" + image_name_nn_t_t1 + ".TIF",
                             W, H,
                             gdal.GDT_Byte, new_anomaly, geo,
                             proj)
    outliers_anomaly.Destroy()
    outliers_t1.Destroy()
    intersection_arr_anomaly_list[t] = ds_anomaly.GetRasterBand(1).ReadAsArray()
    ds_anomaly = None


    #We rewrite tiff with CM with real changes at t:t+1
    ds1 = create_tiff(1, path_results + loss_folder_nn_t_t1 + image_name_nn_t_t1 + ".TIF",
                             W, H,
                             gdal.GDT_Byte, image_array_outliers_nn_t_t1, geo,
                             proj)

    intersection_arr_list[t-1] = ds1.GetRasterBand(1).ReadAsArray()
    ds1 = None



    # we open image t-1 : t
    loss_folder_nn_t_t3, image_name_nn_t_t3, image_array_outliers_nn_t_t3, _, _, _, _, _, _ = open_image(date_prev, date1, path_results)
    image_name_nn_anomaly3 = "Anomaly_" + image_name_nn_t_t3
    outliers_t3 = driver_shp.Open(path_results + loss_folder_nn_t_t3 + image_name_nn_t_t3 + ".shp", 1)
    layer3 = outliers_t3.GetLayer()

    outliers_anomaly3 = driver_shp.Open(path_results + "Anomaly/" + loss_folder_nn_t_t3 + image_name_nn_anomaly3 + ".shp", 1)
    layer_anomaly3 = outliers_anomaly3.GetLayer()


    #we check intersection with previous change map couple and eventually delete the polygons from previous change couple
    for f3 in range(len(layer3)):
        feature3 = layer3[f3]
        id1 = feature3.GetField("value")
        if id1 > 0:
            vectorGeometry3 = feature3.GetGeometryRef()
            x_min, x_max, y_min, y_max, x_res, y_res, xOffset, yOffset = get_bb(vectorGeometry3, pixelWidth, pixelHeight,
                                                                                originX, originY)
            ds_clipped_previous_anomaly = define_extent(layer3, feature3, vectorGeometry3, spatialRef1, x_res, y_res, x_min,
                                               pixelWidth, y_max,
                                               pixelHeight)
            clipped_previous_anomaly = ds_clipped_previous_anomaly.GetRasterBand(1).ReadAsArray()

            clipped_anomaly = new_anomaly[yOffset:yOffset + y_res, xOffset:xOffset + x_res] * clipped_previous_anomaly
            # ds_current_anomaly = gdal.Translate('/vsimem/clip1.tif', ds_anomaly, projWin=[x_min, y_max, x_max, y_min], projWinSRS=proj, outputSRS=proj)
            # clipped_anomaly = ds_current_anomaly.GetRasterBand(1).ReadAsArray()
            # clipped_anomaly = clipped_anomaly * clipped_previous_anomaly
            size_anomaly = np.count_nonzero(clipped_previous_anomaly)


            if (len(np.where((clipped_previous_anomaly.flatten() + clipped_anomaly.flatten()) == 2)[0]) / size_anomaly) >= thr_int:
                #print("Anomaly!!!")
                # it means it's not a change, but an anomaly, so we replace these pixels from normal changes to anomaly
                # we put it in anomaly
                new_previous_anomaly[yOffset:yOffset + y_res, xOffset:xOffset + x_res] = np.copy(new_previous_anomaly[yOffset:yOffset + y_res, xOffset:xOffset + x_res]) + clipped_previous_anomaly
                feature = ogr.Feature(layer3.GetLayerDefn())
                feature.SetField('value', feature3.GetField(0))
                feature.SetGeometry(vectorGeometry3)
                layer_anomaly3.CreateFeature(feature)

                # we delete it from changes
                layer3.DeleteFeature(feature3.GetFID())
                image_array_outliers_nn_t_t3[yOffset:yOffset + y_res, xOffset:xOffset + x_res] = np.copy(image_array_outliers_nn_t_t3[yOffset:yOffset + y_res, xOffset:xOffset + x_res]) - clipped_previous_anomaly
                feature.Destroy()

    new_previous_anomaly[new_previous_anomaly==2] = 1
    image_array_outliers_nn_t_t3[image_array_outliers_nn_t_t3 == 2] = 1
    outliers_anomaly3.Destroy()
    outliers_t3.Destroy()
    #We create tiff with real anomalies
    ds_anomaly = create_tiff(1,
                             path_results + "Anomaly/" + loss_folder_nn_t_t3 + "Anomaly_" + image_name_nn_t_t3 + ".TIF",
                             W, H,
                             gdal.GDT_Byte, new_previous_anomaly, geo,
                             proj)
    intersection_arr_anomaly_list[t-1] = ds_anomaly.GetRasterBand(1).ReadAsArray()
    ds_anomaly = None

    #We rewrite tiff with CM with real changes
    ds3 = create_tiff(1, path_results + loss_folder_nn_t_t3 + image_name_nn_t_t3 + ".TIF",
                             W, H,
                             gdal.GDT_Byte, image_array_outliers_nn_t_t3, geo,
                             proj)

    # We modify the stack of matricies with potential anomalies
    intersection_arr_list[t-1] = ds3.GetRasterBand(1).ReadAsArray()
    ds3 = None
