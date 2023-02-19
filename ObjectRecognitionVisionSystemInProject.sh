
#!/bin/bash
#this application make a photo and after that you can recognize photographed object 
echo "Starting Vision System Application to recognize OralB and Watch"
source bin/activate

#python3 object_detect_sift_3_objects.py
#python3 popUpWindow.py
python3 crop_image_from_camera.py
python3 object_detect_sift_3_objects_mod.py