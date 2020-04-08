This folder contains files to detect bi-temoral changes in the whole SITS.
The user should use three files from this folder:
-   main_pretraining.py - code to pretrain the model
-   main_finetuning.py -  code to fine-tune the model and perform change detection for each couple of images
    We firstly should perform change detection for each couple of t and t+1 images
    In this case in the code you should put:
    image_name1 = os.path.splitext(new_images_list[im])[0]
    image_date1 = list_image_date[im]
    image_name2 = os.path.splitext(new_images_list[im+1])[0]
    image_date2 = list_image_date[im+1]
    Then the segmentation is performed for each couple of t-1 and t+1 images.
    You should put in the code:
    image_name1 = os.path.splitext(new_images_list[im])[0]
    image_date1 = list_image_date[im]
    image_name2 = os.path.splitext(new_images_list[im+2])[0]
    image_date2 = list_image_date[im+2]
    And, of course, you should rename results folder for both cases:
    folder_results = folder_pretrained_results +"t_t1/" + "Joint_AE_"+image_date1 + "_" +image_date2 + "_ep_" + str(epoch_nb) + "_patch_" + str(patch_size) + run_name
    folder_results = folder_pretrained_results +"t_t2/" + "Joint_AE_"+image_date1 + "_" +image_date2 + "_ep_" + str(epoch_nb) + "_patch_" + str(patch_size) + run_name
-	multitemporal_context.py - this script analyses the detected changes in multi-temporal context. 
	It needs two folders t_t1 and t_2,
	the segmentation of the corresponding changes (segmentation scripts in another folder) in vector and raster formats (TIF and shp),
	and the parameter thr_int.
	This code creates a folder with the corrected change detection results with the introduced multi-temporal context.