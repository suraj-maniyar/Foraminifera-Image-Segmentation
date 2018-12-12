# Foraminifera-Image-Segmentation

The code contained in this repository segments the chambers of a Foraminifera image from its edge probability map using Markov Random Field based approach. The original probability map is first refined using morphological transformations and Graph-Cut technique is applied later to perform segmentation. The overall covering score obtained is 71.40%.  

To run the code:  
First run generate_data.py. This will generate the input and output images in the respective folders.  
Now run GraphCut.py. This script will apply graph-cut algorithm on the generated images and save the binary images in image_fine folder.  
Then run morphology.py. This will apply the [Zhang-Suen Thinning Algorithm](https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm) and skeletonize the fine images and save it in image_thin folder.  
Now run watershed.py. This will apply Water-Shed algorithm on the thin skeletonized images and generate the segmented images which are saved in image_segment folder.
