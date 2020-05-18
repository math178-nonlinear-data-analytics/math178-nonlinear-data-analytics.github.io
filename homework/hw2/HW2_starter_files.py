
from pathlib import Path
from typing import List
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

# data pre processing
base_data_folder_path = Path('public_dataset')
file_name_to_colume_names = {
    'Accelerometer.csv': ['Systime', 'EventTime', 'ActivityID', 'X', 'Y', 'Z', 'Phone_orientation'],
    'Activity.csv': ['ID', 'SubjectID', 'Start_time', 'End_time', 'Relative_Start_time', 'Relative_End_time',
                     'Gesture_scenario', 'TaskID', 'ContentID'],
    'Gyroscope.csv': ['Systime', 'EventTime', 'ActivityID', 'X', 'Y', 'Z', 'Phone_orientation'],
}


def get_user_ids() -> List[str]:
    """
    Get all user ids based on name of folders under "public_dataset/"
    :return: a list of user ids
    """
    pass


def get_user_session_ids(user_id: str) -> List[str]:
    """
    Get all session ids for a specific user based on folder structure
    e.g. "public_dataset/100669/100669_session_13" has user_id=100669, session_id=13
    :param user_id: user id
    :return: list of user session ids
    """
    pass

def read_file(user_id: str, user_session_id: str, file_name: str, colume_names: List[str]) -> DataFrame:
    """
    Read one of the csv files for a user
    :param user_id: user id
    :param user_session_id: user session id
    :param file_name: csv file name (key of file_name_to_colume_names)
    :param colume_names: a list of column names of the csv file (value of file_name_to_colume_names)
    :return: content of the csv file as pandas DataFrame
    """
    pass


def get_user_session_data(user_id: str, user_session_id: str) -> DataFrame:
    """
    Combine accelerometer, gyroscope, and activity labels for a specific session of a user
    Note: Timestamps are ignored when joining accelerometer and gyroscope data.
    :param user_id: user id
    :param user_session_id: user session id
    :return: combined DataFrame for a session
    """
    pass

# pick the user as well as activities and extract 3 out of 6 features 
pass
# visualize of the features you pick
pass

def multiV_curvature(nbddata: DataFrame) -> float:  
    """
    Calculate multi V curvature
    :param nbddata: neighborhood of time t_i containing (t, x(t), y(t), z(t)), 
    where x(t), y(t), z(t) are the 3 out of the 6 features. 
    :return: multi V curvature
    """
    pass
def multiV_torsion(nbddata: DataFrame) -> float:  
    """
    Calculate multi V torsion
    :param nbddata: neighborhood of time t_i containing (t, x(t), y(t), z(t)), 
    where x(t), y(t), z(t) are the 3 out of the 6 features. 
    :return: multi V torsion
    """
    pass

# Calucate and plot curvature and torsion of the features you pick
pass

