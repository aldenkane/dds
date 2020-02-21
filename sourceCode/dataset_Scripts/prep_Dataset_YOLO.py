# OptoSwim
# Dataset Engineering Scripts to prep YOLO .txt annotations from LabelMe JSON Files
# Author: Alden Kane
# MIT License

import json
from os import listdir

#######################################################
# Section 1: Declare Globals, Set up Environment
#######################################################

dir_JSON = '../../annotated_100/annotations'
files = listdir(dir_JSON)

#######################################################
# Section 2: Iterate Through Files in Directory and Parse JSON
#######################################################
for file in files:
    # Make sure parsing .json file and ignore hidden files (e.g. .DS_Store)
    if file.endswith('.json'):
        file_Path = str(dir_JSON) + '/' + str(file)
        with open(file_Path, encoding='utf-8-sig') as json_file:
            # Read JSON File
            data = json_file.read()
            data = json.loads(data)
            # Open Text File in same directory as json files
            textFileName = file[:-4] + 'txt'
            txtFile = open(str(dir_JSON) + '/' + str(textFileName),'a')
            # Parse loaded json file
            # Get Image height and width
            imHeight = data['imageHeight']
            imWidth = data['imageWidth']
            annotations = data['shapes']
            for item in annotations:
                objClass = item['label']
                points = item['points']
                if objClass == 'person':
                    #Set Points
                    encObjClass = 0
                    x_Top_Left = points[0][0]
                    y_Top_Left = points[0][1]
                    box_Width = points[1][0]
                    box_Height = points[1][1]
                    scaled_x = x_Top_Left/imWidth
                    scaled_y = y_Top_Left/imHeight
                    scaled_Width = box_Width/imWidth
                    scaled_Height = box_Height/imHeight
                    txtFile.write(str(encObjClass) + ' ' + str(scaled_x) + ' ' + str(scaled_y) + ' ' + str(scaled_Width) + ' ' + str(scaled_Height) + '\n')