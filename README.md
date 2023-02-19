Application used to recognize object using camera, object is photographed and then will be recognize by camera.
I am using OpenCv and Sift library

1. Download and uzip repo

2. Create file visionSystem.sh in the downloaded directory
   touch visionSystem.sh

3. Then run command

`	chmod -R 777 .

4. Open file in the editor and add text below to the file : 


#this application make a photo and seconf part recognize photographed object 
echo "Starting Vision System Application to recognize OralB and Watch"
cd ~
cd Desktop
cd visionSystemObjectRecognition
source bin/activate

python3 crop_image_from_camera.py
python3 object_detect_sift_3_objects_mod.py

5. Save file
6. Run command:
   ./visionSystem.sh
   
 Application should open.
 
 
 #########  OPERATION APP ##############
 
 Three windows should open and camera should be on too. 
 1. In the blue squere put object that you want to recognize later and press space.
 2. Widows will close and next open.
 3. Move object in the front of the camera and it should be recognized.
 In the database are photo for G-Shock watch and Oral-B.
 
 


