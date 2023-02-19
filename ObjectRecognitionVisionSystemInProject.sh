#!/bin/bash
echo "Starting Vision Sytsem Application to recognize OralB and Watch"
cd ~
cd Desktop
cd visionSystemObjectRecognition
source bin/activate

#python3 object_detect_sift_3_objects.py
#python3 popUpWindow.py
python3 crop_image_from_camera.py
python3 object_detect_sift_marek_photo_3_objects_mod.py
