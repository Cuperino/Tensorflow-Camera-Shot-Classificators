#!/usr/bin/python

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        row = {'height': float(root.find('size').find('height').text)}
        flag = {'face':False, 'person':False}
        for member in root.findall('object'):
            if not flag['face'] and (member[0].text=="face" or member[0].text=="face_es"):
                #print("Face")
                row['face_x_min'] = float(member[4][0].text)   # face_x_min
                row['face_y_min'] = float(member[4][1].text)   # face_y_min
                row['face_x_max'] = float(member[4][2].text)   # face_x_max
                row['face_y_max'] = float(member[4][3].text)   # face_y_max
                flag['face'] = True
                # # Set face values in case we forgot to set it separatelly
                row['person_x_min'] = None    # face_x_min
                row['person_y_min'] = None    # face_y_min
                row['person_x_max'] = None    # face_x_max
                row['person_y_max'] = None    # face_y_max
            elif not flag['person'] and member[0].text=="person":
                #print("Person")
                row['person_x_min'] = float(member[4][0].text)   # person_x_min
                row['person_y_min'] = float(member[4][1].text)   # person_y_min
                row['person_x_max'] = float(member[4][2].text)   # person_x_max
                row['person_y_max'] = float(member[4][3].text)   # person_y_max                
                flag['person'] = True
                # If face wasn't previously set,
                if not flag['face']:
                    # Set face values to blank for this row
                    row['face_x_min'] = None    # face_x_min
                    row['face_y_min'] = None    # face_y_min
                    row['face_x_max'] = None    # face_x_max
                    row['face_y_max'] = None    # face_y_max
            elif member[0].text in ('els', 'ls', 'mls', 'ms', 'mcu', 'cu', 'ecu', 'xls', 'xcu'):
                # If set has the wrong syntax by mistake, fix it
                if member[0].text == 'xls':
                    member[0].text = 'els'
                if member[0].text == 'xcu':
                    member[0].text = 'ecu'
                # Set classificator value
                row['class'] = {'els':0, 'ls':1, 'mls':2, 'ms':3, 'mcu':4, 'cu':5, 'ecu':6}[ member[0].text ]
                # Reset flags
                flag['face'] = False
                flag['person'] = False
                if row['face_x_min'] is None or row['person_x_min'] is None:
                    print('Error in file', root.find('filename').text)
                # Save row value
                finalRow = (row['height'], row['face_x_min'], row['face_y_min'], row['face_x_max'], row['face_y_max'], row['person_x_min'], row['person_y_min'], row['person_x_max'], row['person_y_max'], row['class'])
                xml_list.append(finalRow)
                # Continue with next element in loop
                continue
    column_name = [len(xml_list), 9, 'ELS', 'LS', 'MLS', 'MS', 'MCU', 'CU', 'ECU', None, None]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train','test','validate']:
        image_path = os.path.join(os.getcwd(), ('./' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('./' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')

main()
