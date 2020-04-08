
from skimage import filters
from osgeo import gdal
import numpy as np
from codes.image_processing import create_tiff, vectorize_tiff, open_tiff
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import sys
np.set_printoptions(threshold=sys.maxsize)


def histogram(image, nbins=256):
    # For integer types, histogramming with bincount is more efficient.
    if np.issubdtype(image.dtype, np.integer):
        offset = 0
        image_min = np.min(image)
        if image_min < 0:
            offset = image_min
            image_range = np.max(image).astype(np.int64) - image_min
            # get smallest dtype that can hold both minimum and offset maximum
            offset_dtype = np.promote_types(np.min_scalar_type(image_range),
                                            np.min_scalar_type(image_min))
            if image.dtype != offset_dtype:
                # prevent overflow errors when offsetting
                image = image.astype(offset_dtype)
            image = image - offset
        hist = np.bincount(image)
        bin_centers = np.arange(len(hist)) + offset

        # clip histogram to start with a non-zero bin
        idx = np.nonzero(hist)[0][0]
        return hist[idx:], bin_centers[idx:]
    else:
        hist, bin_edges = np.histogram(image.flat, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        return hist, bin_centers

def threshold_otsu(image, nbins=256):
    hist, bin_centers = histogram(image, nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold



def otsu(image_array_loss1, image_array_loss2, H, W, geo, proj, path_results, images_date, threshold=0.995, changes=None, mask=None):
    image_array_loss = np.divide((image_array_loss1+image_array_loss2), 2)

    max_ = np.max(image_array_loss)
    coef = max_/256
    image_array_loss = image_array_loss/coef
    image_array_loss = np.asarray(image_array_loss, dtype=int)
    if mask is not None:
        val = filters.threshold_otsu(np.sort(image_array_loss.flatten()[mask])[0:int(len(mask)*threshold)])
    else:
        val = filters.threshold_otsu(np.sort(image_array_loss.flatten())[0:int(H * W * threshold)])


    image_array_outliers = np.zeros(H*W)
    image_array_outliers[image_array_loss.flatten() > val] = 1
    if mask is not None:
        defected_mask = np.setdiff1d(np.arange(H * W), mask)
        image_array_outliers[defected_mask]=0

    outliers_image_mean = "Outliers_average_" + images_date + "_" +str(threshold)
    dst_ds = create_tiff(1, path_results+ "/"+outliers_image_mean + ".TIF", W, H, gdal.GDT_Byte, np.reshape(image_array_outliers, (H, W)), geo, proj)
    gdal.SieveFilter(dst_ds.GetRasterBand(1), None, dst_ds.GetRasterBand(1), 5, 4)
    dst_ds.FlushCache()
    vectorize_tiff(path_results, "/"+outliers_image_mean, dst_ds)

    dst_ds = None


    if changes is not None:
        if changes in ["changes_2004_2005", "changes_2006_2008"]:
            path_cm = 'C:/Users/Ekaterina_the_Great/Dropbox/IJCNN/images/'+changes
            path_cm = '/home/user/Dropbox/IJCNN/images/' + changes

            path_cm = "/media/user/DATA/Results/RESULTS_CHANGE_DETECTION/GT_Montpellier/"+changes
            cm_truth_name = "mask_changes_small1"
            print(image_array_outliers.shape)
            if changes=="changes_2004_2005":
                cm_predicted = (np.reshape(image_array_outliers, (H, W))[0:600, 600:1400]).flatten()
            if changes == "changes_2006_2008":
                cm_predicted = (np.reshape(image_array_outliers, (H, W))[100:370, 1000:1320]).flatten()
        else:
            if changes in ["changes_Rostov_20150830_20150919", "changes_Rostov_20170918_20180111"]:
                print("hello")
                path_cm = "/media/user/DATA/Results/RESULTS_CHANGE_DETECTION/GT_Rostov/"
                cm_truth_name = changes+"_1"
                if changes == "changes_Rostov_20150830_20150919":
                    print(image_array_outliers.shape)
                    print(np.reshape(image_array_outliers, (H, W)).shape)
                    cm_predicted = (np.reshape(image_array_outliers, (H, W))[0:700, 0:900]).flatten()
                    # cm_predicted = np.asarray(np.reshape(image_array_outliers, (H, W))[0:700, 0:900]).flatten()
                if changes == "changes_Rostov_20170918_20180111":
                    cm_predicted = (np.reshape(image_array_outliers, (H, W))[2100:2400, 900:1400]).flatten()
                cm_predicted[cm_predicted == 0] = 0
                cm_predicted[cm_predicted == 1] = 1


        print(cm_predicted.shape)
        cm_truth, _, _, _, _, _ = open_tiff(path_cm, cm_truth_name)
        cm_truth = cm_truth.flatten()
        cm_truth[cm_truth == 0] = 0
        cm_truth[cm_truth == 1] = 1
        print(cm_truth.shape)
        cm_truth[cm_truth==255]=0
        print(classification_report(cm_truth, cm_predicted, target_names=["no changes", "changes"]))
        print(accuracy_score(cm_truth, cm_predicted))
        print(cohen_kappa_score(cm_truth, cm_predicted))
        conf = confusion_matrix(cm_truth, cm_predicted)
        print(confusion_matrix(cm_truth, cm_predicted))
        omission = conf[1][0]/sum(conf[1])
        print (omission)

