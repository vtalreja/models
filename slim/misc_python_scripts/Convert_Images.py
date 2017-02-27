#import fs
#from PIL import Image

import image_conversion

original_dir=u'/media/veerut/DATADISK/TensorFlow/Iris/CASIA_ND_IRIS_Train_Data'
new_dir=u'/media/veerut/DATADISK/TensorFlow/Iris/CASIA_ND_IRIS_Train_Data_JPG'

image_conversion.convert_dataset_to_png(original_dir, new_dir)
