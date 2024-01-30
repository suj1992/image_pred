import csv
#from pickle import TRUE
import streamlit as st
#rom operator import index
import requests
from exif import Image
import platform
import threading
import time 
#from tkinter import *
#import tkinter as tk
#from tkinter import messagebox
#import mouse
#import datetime
import threading
import numpy as np 
import cv2 
import pyautogui
import sqlite3
import subprocess
#from tkinter.filedialog import askopenfile
#from tkinter import filedialog
#from PIL import ImageTk, Image
import shutil
import pandas as pd
from tqdm import tqdm
import cv2
#import numpy as np
import json
#import shutil
from torch import multiprocessing as mp
import geopandas as gpd
from shapely.geometry import Point
import simplekml
import os

st.title('GEOMATICX MONITERING SYSTEM')

upload = 'uploads'
#output = 'output'

try:
    shutil.rmtree(upload)
    #shutil.rmtree(output)

except:
    pass

# Define the function to save uploaded images
def save_uploaded_images(uploaded_images):
    """
    Save uploaded images to the 'uploads' directory.

    Parameters:
    - uploaded_images (list of file-like objects): List of uploaded image files.

    Returns:
    - success (bool): True if all images were successfully saved, False otherwise.
    """
    success = True  # Assume success initially

    # Create the 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Iterate over each uploaded image file
    for uploaded_image in uploaded_images:
        try:
            # Save the image file to the 'uploads' directory
            with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
                f.write(uploaded_image.getbuffer())
        except Exception as e:
            # If an error occurs while saving the image, set success to False
            success = False
            st.error(f"Error saving {uploaded_image.name}: {str(e)}")

    return success

# Main Streamlit code
def main():
    st.title("Upload Images")

    # Allow users to upload multiple image files
    uploaded_images = st.file_uploader("Upload multiple image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # Check if files are uploaded
    if uploaded_images:
        # Save the uploaded images
        if save_uploaded_images(uploaded_images):
            st.success("All images uploaded successfully and saved.")
        else:
            st.error("Failed to save one or more images.")
if __name__ == "__main__":
    main()



img_path = 'uploads' 
def pole_cross_arm(img_path):
    
    import os
    
    # Specify the path to the directory you want to delete
    directory_path_1 = 'yolov5/runs/detect/exp'
    directory_path_2 = 'yolov5/runs/detect/exp2'
    directory_path_3 = 'yolov5/runs/detect/exp3'
    directory_path_4 = 'yolov5/runs/detect/exp4'
    directory_path_5 = 'yolov5/runs/detect/exp5'

    try:
        # Delete the directory and its contents
        shutil.rmtree(directory_path_1)
        shutil.rmtree(directory_path_2)
        shutil.rmtree(directory_path_3)
        shutil.rmtree(directory_path_4)
        shutil.rmtree(directory_path_5)
        #print(f"Directory '{directory_path}' deleted successfully.")
    except:
         pass
    

#------------------------------------------------LAT LONG EXTRACTION----------------------------------------------------------------
    from exif import Image
    try:


        def decimal_coords(coords, ref):
            decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
            if ref == "S" or ref =='W' :
                decimal_degrees = -decimal_degrees
            return decimal_degrees

        def image_coordinates(image_path):

            with open(image_path, 'rb') as src:
                img = Image(src)
            if img.has_exif:
                try:
                    img.gps_longitude
                    coords = (decimal_coords(img.gps_latitude,
                            img.gps_latitude_ref),
                            decimal_coords(img.gps_longitude,
                            img.gps_longitude_ref))
                except:
                    return ('NX')
            else:
                pass
                
            return({coords[1],coords[0]})# coords[0]= latitude, coords[1]=longitude
        
        image_filenames = []
        co_ordinates = []

        if os.listdir(img_path):
            # Iterate through each YOLO output text file
            for filename in os.listdir(img_path):
                path = os.path.join(img_path, filename)
                #print(path)
                # Call the function to extract coordinates from the text file
                co_or = image_coordinates(path)
                image_filenames.append(path)
                co_ordinates.append(co_or)

        # Create a DataFrame from the list of coordinates and filenames
        df_cord = pd.DataFrame(list(zip(image_filenames, co_ordinates)), columns=['File Name', 'Coordinates'])

        for i in df_cord.index:
            df_cord['Coordinates'][i] = list(df_cord['Coordinates'][i])

        df_cord['lat'] = ' '
        df_cord['lng'] = ' '

        for i in df_cord.index:
            df_cord['lng'][i] = df_cord['Coordinates'][i][0]
            df_cord['lat'][i] = df_cord['Coordinates'][i][1]

        df_cord = df_cord.drop(columns={'Coordinates'})
        for i in df_cord.index:
            if df_cord['lng'][i] == 'N' and df_cord['lat'][i] == 'X':
                df_cord['lat'][i] = ' '
                df_cord['lng'][i] = ' '
            else:
                pass
        #print(df_cord)
    except:
        pass
#----------------------------------------------------DIRECTORY PATH CREATION--------------------------------------------------
    directory_path_1 = "output"
    directory_path_2 = "output/static_image"
    directory_path_3 = "output/shape_output"
    
    os.makedirs(directory_path_1, exist_ok=True)
    os.makedirs(directory_path_2, exist_ok=True)
    os.makedirs(directory_path_3, exist_ok=True)
    
    
   # Extract values
    def extract_first_values(data):
        try:
            lines = data.strip().split('\n')  # Split the data into lines
            first_values = [line.split()[0] for line in lines]  # Extract the first values from each line
            return first_values
        except:
            pass

#-----------------------------------------------------------ALL MODEL----------------------------------------------------------
    os.system(f"python yolov5/detect.py --weights yolov5/runs/train/pole_cross_arm_drone/weights/best.pt --img 640 --conf 0.25 --iou-thres 0.10 --source {img_path} --line-thickness 1 --save-txt")
    #os.system(f"python yolov5/detect.py --weights yolov5/runs/train/pole_cross_arm_drone/weights/best.pt --img 640 --conf 0.25 --iou-thres 0.10 --source {img_path} --line-thickness 1 --save-txt")
    os.system(f"python yolov5/detect.py --weights yolov5/runs/train/pole_material/weights/best.pt --img 640 --conf 0.25 --iou-thres 0.10 --source {img_path} --line-thickness 1 --save-txt")
    os.system(f"python yolov5/detect.py --weights yolov5/runs/train/Insulator_type/weights/best.pt --img 640 --conf 0.25 --iou-thres 0.10 --source {img_path} --line-thickness 1 --save-txt")
    os.system(f"python yolov5/detect.py --weights yolov5/runs/train/Insulator_material/weights/best.pt --img 640 --conf 0.25 --iou-thres 0.10 --source {img_path} --line-thickness 1 --save-txt")
    os.system(f"python yolov5/detect.py --weights yolov5/runs/train/street_light/weights/best.pt --img 640 --conf 0.25 --iou-thres 0.10 --source {img_path} --line-thickness 1 --save-txt")
#----------------------------------------------------------------------------------------------------------------------------------
    

        
#-----------------------------------------------------------POLE CROSS ARM---------------------------------------------------------   
    # Define the path to the directory containing YOLO output text files
    output_dir = "yolov5/runs/detect/exp/labels"
    output_dir_1 = "yolov5/runs/detect/exp"
    
    #print("************",os.listdir(output_dir))

    # Initialize lists to store image filenames and confidence scores
    image_filenames_1 = []
    confidence_scores_1 = []
    #print(img_path.split('/')[-1])

    for filename in os.listdir(output_dir_1):
        if filename.endswith(".jpg"):
            image_filename = os.path.splitext(filename)[0] + ".txt"
            image_filenames_1.append(image_filename)
            confidence_scores_1.append(9)

    df_full = pd.DataFrame({"File Name": image_filenames_1, "cross_arm": confidence_scores_1})
    #print("**/*/*/*/*/*/*/*/*",df_full)
    
    
    
    image_filenames = []
    confidence_scores = []
    if os.listdir(output_dir):
    # Iterate through each YOLO output text file
        for filename in os.listdir(output_dir):
            if filename.endswith(".txt"):
                image_filename = os.path.splitext(filename)[0] + ".txt"  # Assuming image files have a ".jpg" extension
                with open(os.path.join(output_dir, filename), "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        # Split the line by space
                        parts = line.strip().split(" ")
                        if len(parts) >= 2:
                            confidence = float(parts[1])  # Confidence score
                            image_filenames.append(image_filename)
                            confidence_scores.append(confidence)

        # Create a DataFrame from image filenames and confidence scores
        df_confidence = pd.DataFrame({"File Name": image_filenames, "con_cross_arm": confidence_scores})

        label_directory = "yolov5/runs/detect/exp/labels"

        label_files = [f for f in os.listdir(label_directory) if f.endswith(".txt")]

        data = {"File Name": [], "Label Data": []}

        for label_file in label_files:
            with open(os.path.join(label_directory, label_file), "r") as file:
                label_data = file.read()
                data["File Name"].append(label_file)  # Store the file name
                data["Label Data"].append(label_data)  # Store the label data

        df_label = pd.DataFrame(data)

        # Display the DataFrame
        df_label

        df_lab_con = pd.merge(df_confidence, df_label, on = 'File Name', how = 'outer')
    
        def extract_info(row):
            #numeric_value = int(row['File Name'].split('.')[0])
            
            # Check if 'Label Data' is a string and not a float
            if isinstance(row['Label Data'], str):
                label_1 = row['Label Data'][0]
            else:
                label_1 = str(row['Label Data'])
            
            return pd.Series({'cross_arm': label_1})

        # Apply the function to each row in the DataFrame and assign the result
        df_lab_con['cross_arm'] = df_lab_con.apply(extract_info, axis=1)

        # Drop Duplicate Value in 'Numeric Value' column
        df_lab_con = df_lab_con.drop_duplicates(subset='File Name')

        # Drop rows with missing values (NaN)
        df_lab_con.dropna(inplace=True)
        df_lab_con.rename(columns={'Label Data': 'Label_Crs_Arm'}, inplace=True)
        #print(df_lab_con)
        df_lab_con = pd.concat([df_lab_con, df_full]).drop_duplicates(subset='File Name', keep='first')
        '''name = img_path.split('/')[-1]
        df_lab_con['File Name'] = img_path.split('/')[-1]
        print(df_lab_con)'''
    else:
        pass


         
    csv_path='output/static_image/cross_arm_pred.csv'
    df_lab_con.to_csv(csv_path, index=False)
         
#----------------------------------------------POLE MATERIAL---------------------------------------------------------------------
    # Define the path to the directory containing YOLO output text files
    output_dir = "yolov5/runs/detect/exp2/labels"
    #print("************",os.listdir(output_dir))
        # Initialize lists to store image filenames and confidence scores
    image_filenames_2 = []
    confidence_scores_2 = []
    #print(img_path.split('/')[-1])

    for filename in os.listdir(output_dir_1):
        if filename.endswith(".jpg"):
            image_filename = os.path.splitext(filename)[0] + ".txt"
            image_filenames_2.append(image_filename)
            confidence_scores_2.append(9)

    df_full = pd.DataFrame({"File Name": image_filenames_2, "pl_mat": confidence_scores_2})

    # Initialize lists to store image filenames and confidence scores
    image_filenames = []
    confidence_scores = []
    #print(img_path.split('/')[-1])

    if os.listdir(output_dir):
    # Iterate through each YOLO output text file
        for filename in os.listdir(output_dir):
            if filename.endswith(".txt"):
                image_filename = os.path.splitext(filename)[0] + ".txt"  # Assuming image files have a ".jpg" extension
                with open(os.path.join(output_dir, filename), "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        # Split the line by space
                        parts = line.strip().split(" ")
                        if len(parts) >= 2:
                            confidence = float(parts[1])  # Confidence score
                            image_filenames.append(image_filename)
                            confidence_scores.append(confidence)

        # Create a DataFrame from image filenames and confidence scores
        df_confidence = pd.DataFrame({"File Name": image_filenames, "con_pl_mat": confidence_scores})

        label_directory = "yolov5/runs/detect/exp2/labels"

        label_files = [f for f in os.listdir(label_directory) if f.endswith(".txt")]

        data = {"File Name": [], "Label Data": []}

        for label_file in label_files:
            with open(os.path.join(label_directory, label_file), "r") as file:
                label_data = file.read()
                data["File Name"].append(label_file)  # Store the file name
                data["Label Data"].append(label_data)  # Store the label data

        df_label = pd.DataFrame(data)

        # Display the DataFrame
        df_label

        df_lab_con = pd.merge(df_confidence, df_label, on = 'File Name', how = 'outer')


        def extract_info(row):
            #numeric_value = int(row['File Name'].split('.')[0])
            
            # Check if 'Label Data' is a string and not a float
            if isinstance(row['Label Data'], str):
                label_1 = row['Label Data'][0]
            else:
                label_1 = str(row['Label Data'])
            
            return pd.Series({'pl_mat': label_1})

        # Apply the function to each row in the DataFrame and assign the result
        df_lab_con['pl_mat'] = df_lab_con.apply(extract_info, axis=1)

        # Drop Duplicate Value in 'Numeric Value' column
        df_lab_con = df_lab_con.drop_duplicates(subset='File Name')

        # Drop rows with missing values (NaN)
        df_lab_con.dropna(inplace=True)
        df_lab_con.rename(columns={'Label Data': 'Label_Pl_mat'}, inplace=True)
        #print(df_lab_con)
        '''name = img_path.split('/')[-1]
        df_lab_con['File Name'] = name = img_path.split('/')[-1]'''
        df_lab_con = pd.concat([df_lab_con, df_full]).drop_duplicates(subset='File Name', keep='first')
        #print(df_lab_con)
    else:
         df_lab_con = df_full
         '''name = img_path.split('/')[-1]
         print(name)
         data = {'File Name': [name],
        'pl_mat': ['Null']}
         df_lab_con = pd.DataFrame(data)
         print(df_lab_con)'''
         
    #df_lab_con['pl_mat'] = df_lab_con['pl_mat'].apply(categorize_list_pl_mat)
	
    csv_path='output/static_image/pl_mat_pred.csv'
    df_lab_con.to_csv(csv_path, index=False)
    
#----------------------------------------------------INSULATOR TYPE---------------------------------------------------------------
    
    # Define the path to the directory containing YOLO output text files
    output_dir = "yolov5/runs/detect/exp3/labels"
    #print("************",os.listdir(output_dir))
    image_filenames_3 = []
    confidence_scores_3 = []
    #print(img_path.split('/')[-1])

    for filename in os.listdir(output_dir_1):
        if filename.endswith(".jpg"):
            image_filename = os.path.splitext(filename)[0] + ".txt"
            image_filenames_3.append(image_filename)
            confidence_scores_3.append(9)

    df_full = pd.DataFrame({"File Name": image_filenames_3, "ins_typ": confidence_scores_3})

    # Initialize lists to store image filenames and confidence scores
    image_filenames = []
    confidence_scores = []
    #print(img_path.split('/')[-1])

    if os.listdir(output_dir):
    # Iterate through each YOLO output text file
        for filename in os.listdir(output_dir):
            if filename.endswith(".txt"):
                image_filename = os.path.splitext(filename)[0] + ".txt"  # Assuming image files have a ".jpg" extension
                with open(os.path.join(output_dir, filename), "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        # Split the line by space
                        parts = line.strip().split(" ")
                        if len(parts) >= 2:
                            confidence = float(parts[1])  # Confidence score
                            image_filenames.append(image_filename)
                            confidence_scores.append(confidence)

        # Create a DataFrame from image filenames and confidence scores
        df_confidence = pd.DataFrame({"File Name": image_filenames, "con_ins_typ": confidence_scores})

        label_directory = "yolov5/runs/detect/exp3/labels"

        label_files = [f for f in os.listdir(label_directory) if f.endswith(".txt")]

        data = {"File Name": [], "Label Data": []}

        for label_file in label_files:
            with open(os.path.join(label_directory, label_file), "r") as file:
                label_data = file.read()
                data["File Name"].append(label_file)  # Store the file name
                data["Label Data"].append(label_data)  # Store the label data

        df_label = pd.DataFrame(data)

        # Display the DataFrame
        df_label

        df_lab_con = pd.merge(df_confidence, df_label, on = 'File Name', how = 'outer')


        def extract_info(row):
            #numeric_value = int(row['File Name'].split('.')[0])
            
            # Check if 'Label Data' is a string and not a float
            if isinstance(row['Label Data'], str):
                label_1 = row['Label Data'][0]
            else:
                label_1 = str(row['Label Data'])
            
            return pd.Series({'ins_typ': label_1})

        # Apply the function to each row in the DataFrame and assign the result
        df_lab_con['ins_typ'] = df_lab_con.apply(extract_info, axis=1)

        # Drop Duplicate Value in 'Numeric Value' column
        df_lab_con = df_lab_con.drop_duplicates(subset='File Name')

        # Drop rows with missing values (NaN)
        df_lab_con.dropna(inplace=True)
        df_lab_con.rename(columns={'Label Data': 'Label_ins_typ'}, inplace=True)
        #print(df_lab_con)
        '''name = img_path.split('/')[-1]
        df_lab_con['File Name'] = name = img_path.split('/')[-1]'''
        df_lab_con = pd.concat([df_lab_con, df_full]).drop_duplicates(subset='File Name', keep='first')
        #print(df_lab_con)
    else:
         df_lab_con = df_full
         '''name = img_path.split('/')[-1]
         print(name)
         data = {'File Name': [name],
        'ins_typ': ['Null']}
         df_lab_con = pd.DataFrame(data)
         print(df_lab_con)'''
         
    #df_lab_con['ins_typ'] = df_lab_con['ins_typ'].apply(categorize_list_pl_mat)
	
    csv_path='output/static_image/ins_typ_pred.csv'
    df_lab_con.to_csv(csv_path, index=False)
    

#--------------------------------------------------------------INSULATOR MATERIAL---------------------------------------------------
    # Define the path to the directory containing YOLO output text files
    output_dir = "yolov5/runs/detect/exp4/labels"
    #print("************",os.listdir(output_dir))

    image_filenames_4 = []
    confidence_scores_4 = []
    #print(img_path.split('/')[-1])

    for filename in os.listdir(output_dir_1):
        if filename.endswith(".jpg"):
            image_filename = os.path.splitext(filename)[0] + ".txt"
            image_filenames_4.append(image_filename)
            confidence_scores_4.append(9)

    df_full = pd.DataFrame({"File Name": image_filenames_4, "ins_mat": confidence_scores_4})

    # Initialize lists to store image filenames and confidence scores
    image_filenames = []
    confidence_scores = []
    #print(img_path.split('/')[-1])

    if os.listdir(output_dir):
    # Iterate through each YOLO output text file
        for filename in os.listdir(output_dir):
            if filename.endswith(".txt"):
                image_filename = os.path.splitext(filename)[0] + ".txt"  # Assuming image files have a ".jpg" extension
                with open(os.path.join(output_dir, filename), "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        # Split the line by space
                        parts = line.strip().split(" ")
                        if len(parts) >= 2:
                            confidence = float(parts[1])  # Confidence score
                            image_filenames.append(image_filename)
                            confidence_scores.append(confidence)

        # Create a DataFrame from image filenames and confidence scores
        df_confidence = pd.DataFrame({"File Name": image_filenames, "con_ins_mat": confidence_scores})

        label_directory = "yolov5/runs/detect/exp4/labels"

        label_files = [f for f in os.listdir(label_directory) if f.endswith(".txt")]

        data = {"File Name": [], "Label Data": []}

        for label_file in label_files:
            with open(os.path.join(label_directory, label_file), "r") as file:
                label_data = file.read()
                data["File Name"].append(label_file)  # Store the file name
                data["Label Data"].append(label_data)  # Store the label data

        df_label = pd.DataFrame(data)

        # Display the DataFrame
        df_label

        df_lab_con = pd.merge(df_confidence, df_label, on = 'File Name', how = 'outer')


        def extract_info(row):
            #numeric_value = int(row['File Name'].split('.')[0])
            
            # Check if 'Label Data' is a string and not a float
            if isinstance(row['Label Data'], str):
                label_1 = row['Label Data'][0]
            else:
                label_1 = str(row['Label Data'])
            
            return pd.Series({'ins_mat': label_1})

        # Apply the function to each row in the DataFrame and assign the result
        df_lab_con['ins_mat'] = df_lab_con.apply(extract_info, axis=1)

        # Drop Duplicate Value in 'Numeric Value' column
        df_lab_con = df_lab_con.drop_duplicates(subset='File Name')

        # Drop rows with missing values (NaN)
        df_lab_con.dropna(inplace=True)
        df_lab_con.rename(columns={'Label Data': 'Label_ins_mat'}, inplace=True)
        #print(df_lab_con)
        '''name = img_path.split('/')[-1]
        df_lab_con['File Name'] = name = img_path.split('/')[-1]'''
        df_lab_con = pd.concat([df_lab_con, df_full]).drop_duplicates(subset='File Name', keep='first')
        #print(df_lab_con)
    else:
         df_lab_con = df_full
         '''name = img_path.split('/')[-1]
         print(name)
         data = {'File Name': [name],
        'ins_mat': ['Null']}
         df_lab_con = pd.DataFrame(data)
         print(df_lab_con)'''
         
    #df_lab_con['ins_mat'] = df_lab_con['ins_mat'].apply(categorize_list_pl_mat)
	
    csv_path='output/static_image/ins_mat_pred.csv'
    df_lab_con.to_csv(csv_path, index=False)
    
#--------------------------------------------------STREET LIGHT--------------------------------------------------------------
    # Define the path to the directory containing YOLO output text files
    output_dir = "yolov5/runs/detect/exp5/labels"
    #print("************",os.listdir(output_dir))

    image_filenames_5 = []
    confidence_scores_5 = []
    #print(img_path.split('/')[-1])

    for filename in os.listdir(output_dir_1):
        if filename.endswith(".jpg"):
            image_filename = os.path.splitext(filename)[0] + ".txt"
            image_filenames_5.append(image_filename)
            confidence_scores_5.append(9)

    df_full = pd.DataFrame({"File Name": image_filenames_5, "street_lgt": confidence_scores_5})
    # Initialize lists to store image filenames and confidence scores
    image_filenames = []
    confidence_scores = []
    #print(img_path.split('/')[-1])

    if os.listdir(output_dir):
    # Iterate through each YOLO output text file
        for filename in os.listdir(output_dir):
            if filename.endswith(".txt"):
                image_filename = os.path.splitext(filename)[0] + ".txt"  # Assuming image files have a ".jpg" extension
                with open(os.path.join(output_dir, filename), "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        # Split the line by space
                        parts = line.strip().split(" ")
                        if len(parts) >= 2:
                            confidence = float(parts[1])  # Confidence score
                            image_filenames.append(image_filename)
                            confidence_scores.append(confidence)

        # Create a DataFrame from image filenames and confidence scores
        df_confidence = pd.DataFrame({"File Name": image_filenames, "con_strt_lgt": confidence_scores})

        label_directory = "yolov5/runs/detect/exp5/labels"

        label_files = [f for f in os.listdir(label_directory) if f.endswith(".txt")]

        data = {"File Name": [], "Label Data": []}

        for label_file in label_files:
            with open(os.path.join(label_directory, label_file), "r") as file:
                label_data = file.read()
                data["File Name"].append(label_file)  # Store the file name
                data["Label Data"].append(label_data)  # Store the label data

        df_label = pd.DataFrame(data)

        # Display the DataFrame
        df_label

        df_lab_con = pd.merge(df_confidence, df_label, on = 'File Name', how = 'outer')


        def extract_info(row):
            #numeric_value = int(row['File Name'].split('.')[0])
            
            # Check if 'Label Data' is a string and not a float
            if isinstance(row['Label Data'], str):
                label_1 = row['Label Data'][0]
            else:
                label_1 = str(row['Label Data'])
            
            return pd.Series({'street_lgt': label_1})

        # Apply the function to each row in the DataFrame and assign the result
        df_lab_con['street_lgt'] = df_lab_con.apply(extract_info, axis=1)

        # Drop Duplicate Value in 'Numeric Value' column
        df_lab_con = df_lab_con.drop_duplicates(subset='File Name')

        # Drop rows with missing values (NaN)
        df_lab_con.dropna(inplace=True)
        df_lab_con.rename(columns={'Label Data': 'Label_strt_lgt'}, inplace=True)
        #print(df_lab_con)
        '''name = img_path.split('/')[-1]
        df_lab_con['File Name'] = name = img_path.split('/')[-1]'''
        df_lab_con = pd.concat([df_lab_con, df_full]).drop_duplicates(subset='File Name', keep='first')
        #print(df_lab_con)
    else:
         df_lab_con = df_full
         '''name = img_path.split('/')[-1]
         print(name)
         data = {'File Name': [name],
        'street_lgt': ['Null']}
         df_lab_con = pd.DataFrame(data)
         print(df_lab_con)'''
         
    #df_lab_con['street_lgt'] = df_lab_con['street_lgt'].apply(categorize_list_pl_mat)
	
    csv_path='output/static_image/street_lgt_pred.csv'
    df_lab_con.to_csv(csv_path, index=False)
    

#---------------------------------------FUNCTION FOR VALUES OF ATTRIBUTES------------------------------------------------------


#-------------------------------------- READ ALL FILES----------------------------------------------------------

    df_crss_arm =  pd.read_csv('output/static_image/cross_arm_pred.csv')
    df_ins_mat = pd.read_csv('output/static_image/ins_mat_pred.csv')
    df_ins_typ = pd.read_csv('output/static_image/ins_typ_pred.csv')
    df_pl_mat = pd.read_csv('output/static_image/pl_mat_pred.csv')
    df_st_lgt = pd.read_csv('output/static_image/street_lgt_pred.csv')

    #Merge all data frame
    df_1 = pd.merge(df_crss_arm, df_ins_mat, on = ['File Name'], how = 'outer')
    df_2 = pd.merge(df_1, df_ins_typ, on = ['File Name'], how = 'outer')
    df_3 = pd.merge(df_2, df_pl_mat, on = ['File Name'], how = 'outer')
    df_4 = pd.merge(df_3, df_st_lgt, on = ['File Name'], how = 'outer')
    #print("////////////******************////////",df_4['Label_ins_mat'][0][0])
    




    #df_4['cross_arm'] = df_4['Label_Crs_Arm'].apply(extract_first_values)
    

    # pole cross arm
        
    '''df_4['cross_arm'] = df_4['cross_arm'].apply(categorize_list_pl_crs_arm)
    df_4['ins_mat'] = df_4['ins_mat'].apply(categorize_list_ins_mat)
    df_4['ins_typ'] = df_4['ins_typ'].apply(categorize_list_ins_typ)
    df_4['[pl_mat]'] = df_4['pl_mat'].apply(categorize_list_pl_mat)
    df_4['street_lgt'] = df_4['street_lgt'].apply(categorize_street_lgt_typ)
    #df_4['cross_arm'] = df_4['cross_arm'].apply(categorize_list_pl_crs_arm)
    print("**//**//**//",df_4)'''
    csv_path='output/static_image/output.csv'
    df_4.to_csv(csv_path, index=False)
    df=  pd.read_csv('output/static_image/output.csv')
    print(df)

    for i in df.index:
        if df['cross_arm'][i] == 0:
            df['cross_arm'][i] = 'V Cross arm'
        elif df['cross_arm'][i] == 1:
             df['cross_arm'][i] = 'Mixed'
        elif df['cross_arm'][i] == 2:
            df['cross_arm'][i] = 'Cantilever'
        elif df['cross_arm'][i] ==  3:
            df['cross_arm'][i] = 'Horizontal Cross Arm'
        else:
           df['cross_arm'][i] = 'Unknown'
           
    for i in df.index:
        if df['pl_mat'][i] == 0:
            df['pl_mat'][i] = 'Concrete'
        elif df['pl_mat'][i] == 1:
             df['pl_mat'][i] = 'GI'
        elif df['pl_mat'][i] == 2:
            df['pl_mat'][i] = 'RSJ'
        elif df['pl_mat'][i] ==  3:
            df['pl_mat'][i] = 'Rail'
        elif df['pl_mat'][i] ==  4:
            df['pl_mat'][i] = 'Wood'
        elif df['pl_mat'][i] ==  5:
            df['pl_mat'][i] = 'Steel Tabular'
        else:
           df['pl_mat'][i] = 'Unknown'
           
    for i in df.index:
        if df['ins_mat'][i] == 0:
            df['ins_mat'][i] = 'Polymer'
        elif df['ins_mat'][i] == 1:
             df['ins_mat'][i] = 'Porcelain'
        elif df['ins_mat'][i] == 0 and df['ins_mat'][i] == 1:
            df['ins_mat'][i] = 'Polymer+Porcelain'
        elif df['ins_mat'][i] ==  2:
            df['ins_mat'][i] = 'Unknown'
        else:
           df['ins_mat'][i] = 'Unknown'

    for i in df.index:
        if df['ins_typ'][i] == 0:
            df['ins_typ'][i] = 'Pin'
        elif df['ins_typ'][i] == 1:
             df['ins_typ'][i] = 'Shackle'
        elif df['ins_typ'][i] == 3:
            df['ins_typ'][i] = 'Unknown'
        elif df['ins_typ'][i] ==  2:
            df['ins_typ'][i] = 'Disc'
        elif df['ins_typ'][i] ==  0 and df['ins_typ'][i] ==  1:
            df['ins_typ'][i] = 'Pin+Shackle'
        elif df['ins_typ'][i] ==  0 and df['ins_typ'][i] ==  2:
            df['ins_typ'][i] = 'Pin+Disc'
        else:
           df['ins_typ'][i] = 'Unknown'      

    for i in df.index:
        if df['street_lgt'][i] == 0:
            df['street_lgt'][i] = 'Blub'
        elif df['street_lgt'][i] == 1:
             df['street_lgt'][i] = 'CLF'
        elif df['street_lgt'][i] == 2:
            df['street_lgt'][i] = 'LED'
        elif df['street_lgt'][i] ==  3:
            df['street_lgt'][i] = 'Tubelight'
        elif df['street_lgt'][i] == 4:
            df['street_lgt'][i] = 'HPSV'
        elif df['street_lgt'][i] ==  5:
            df['street_lgt'][i] = 'HPMV'
        else:
           df['street_lgt'][i] = 'Unknown' 



    for i in df.index:
        #print(df['File Name'][i].split('.'))
        df['File Name'][i] = df['File Name'][i].split('.')[0]
        df.at[i,'File Name'] = f'file://{img_path}/{df.at[i, "File Name"]}.jpg'

    try:
        for i in df_cord.index:
            df_cord.at[i,'File Name'] = f'file://{df_cord.at[i, "File Name"]}'
        
        df = pd.merge(df, df_cord, on=['File Name'], how='outer')
    except:
        pass
    '''def create_hyperlink(row):
        file_name = row['File Name']
        #path_prefix = row['Path Prefix']
        hyperlink = f'HYPERLINK("{file_name}")'
        return hyperlink
    
    df['File Name'] = df.apply(create_hyperlink, axis=1)'''

        
    csv_path='output/static_image/output_final.csv'
    df.to_csv(csv_path, index=False)

#df.at[i, 'File Name'] = 'file://' + img_path + '/' + df.at[i, 'File Name']
    df['lat'] = df['lat'].replace(' ', np.nan)

# Convert whitespace values to NaN in 'lng' column
    df['lng'] = df['lng'].replace(' ', np.nan)

# Drop rows with missing latitude or longitude values
    df_shp = df.dropna(subset=['lat', 'lng'])
    #print('*************////*****',df_shp)
#---------------------------------------------------SHAPE FILE CREATION------------------------------------------------------------
    try:
        geometry = [Point(lon, lat) for lon, lat in zip(df_shp['lng'], df_shp['lat'])]

        df_new_1 = gpd.GeoDataFrame(df_shp, geometry=geometry)


        output_path = 'output/shape_output/pole.shp' 

        df_new_1.to_file(output_path, driver='ESRI Shapefile')    
    except:
        pass

        
#-------------------------------------------------------KML FILE--------------------------------------------------------------------
    try:
        kml = simplekml.Kml()
        for index, row in df.iterrows():
            name = row['File Name'].split('/')[-1]
            cross_arm = row['cross_arm']
            ins_mat = row['ins_mat']
            ins_typ =  row['ins_typ']
            pl_mat =  row['pl_mat']
            street_lgt =  row['street_lgt']
            image_name = row['File Name']
            
            lat = row['lat']
            lon = row['lng']
            pnt = kml.newpoint(name=name, coords=[(lon, lat)])
            pnt.extendeddata.newdata(name="cross_arm", value=cross_arm)
            pnt.extendeddata.newdata(name="ins_mat", value=ins_mat)
            pnt.extendeddata.newdata(name="ins_typ", value=ins_typ)
            pnt.extendeddata.newdata(name="pl_mat", value=pl_mat)
            pnt.extendeddata.newdata(name="street_lgt", value=street_lgt)
            pnt.extendeddata.newdata(name="image_name", value=image_name)

        kml_path = 'output/pole.kml'
        kml.save(kml_path)
    except:
        pass





if st.button('Pridict'):
    pole_cross_arm(img_path)
###------------------------------------------------------KML DOWNLOAD--------------------------------------------------------------
kml_path = 'output/pole.kml'
def download_kml_file(kml_path):
    with open(kml_path, 'rb') as f:
        kml_data = f.read()
    return kml_data

# Main Streamlit code
def main():
    st.title("Download KML File")

    # Button to download the KML file
    if st.button("Download KML File"):
        kml_bytes = download_kml_file(kml_path)
        st.download_button(label="Click here to download", data=kml_bytes, file_name=os.path.basename(kml_path))

if __name__ == "__main__":
    main()

#----------------------------------------------CSV DOWNLOAD-----------------------------------------------------------------
csv_path = 'output/static_image/output_final.csv'

def download_csv_file(csv_path):
    with open(csv_path, 'rb') as f:
        csv_data = f.read()
    return csv_data

# Main Streamlit code
def main():
    st.title("Final Output CSV")

    # Button to download the KML file
    if st.button("Final Output CSV"):
        csv_bytes = download_csv_file(csv_path)
        st.download_button(label="Click here to download", data=csv_bytes, file_name=os.path.basename(csv_path))

if __name__ == "__main__":
    main()


#--------------------------------------------------------SHAPE DOWNLOAD-------------------------------------------------------
    
import streamlit as st
import os
import shutil
import zipfile

# Define the path to the folder you want to download
folder_path = 'output/shape_output'

# Function to create a zip file from the folder contents
def create_zip(folder_path, zip_file_name):
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))

# Main Streamlit code
def main():
    st.title("Download Folder as Zip")

    # Button to download the folder as a zip file
    if st.button("Download Folder"):
        # Temporary zip file name
        temp_zip_file = 'temp_folder.zip'

        # Create a zip file containing the folder contents
        create_zip(folder_path, temp_zip_file)

        # Download the zip file
        with open(temp_zip_file, 'rb') as f:
            zip_data = f.read()
        st.download_button(label="Click here to download", data=zip_data, file_name='shape_output.zip', mime='application/zip')

        # Delete the temporary zip file
        os.remove(temp_zip_file)

if __name__ == "__main__":
    main()
