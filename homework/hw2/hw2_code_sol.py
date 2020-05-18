
from pathlib import Path
from typing import List
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

base_data_folder_path = Path('public_dataset')
file_name_to_colume_names = {
    'Accelerometer.csv': ['Systime', 'EventTime', 'ActivityID', 'X', 'Y', 'Z', 'Phone_orientation'],
    'Activity.csv': ['ID', 'SubjectID', 'Start_time', 'End_time', 'Relative_Start_time', 'Relative_End_time', 'Gesture_scenario', 'TaskID', 'ContentID'],
    'Gyroscope.csv': ['Systime', 'EventTime', 'ActivityID', 'X', 'Y', 'Z', 'Phone_orientation'],
}
def get_user_ids() -> List[str]:
    """
    Get all user ids based on name of folders under "public_dataset/"
    :return: a list of user ids
    """
    user_ids = [file_path.name for file_path in base_data_folder_path.iterdir() if file_path.is_dir()]
    return user_ids
def get_user_session_ids(user_id: str) -> List[str]:
    """
    Get all session ids for a specific user based on folder structure
    e.g. "public_dataset/100669/100669_session_13" has user_id=100669, session_id=13
    :param user_id: user id
    :return: list of user session ids
    """
    user_folder_path = base_data_folder_path / str(user_id)
    user_session_folder_names = [file_path.name for file_path in user_folder_path.iterdir() if file_path.is_dir()]
    user_session_ids = [user_session_folder_name.split('_')[2] for user_session_folder_name in user_session_folder_names]
    return user_session_ids
def read_file(user_id: str, user_session_id: str, file_name: str, colume_names: List[str]) -> DataFrame:
    """
    Read one of the csv files for a user
    :param user_id: user id
    :param user_session_id: user session id
    :param file_name: csv file name (key of file_name_to_colume_names)
    :param colume_names: a list of column names of the csv file (value of file_name_to_colume_names)
    :return: content of the csv file as pandas DataFrame
    """
    session_folder_name = user_id + '_session_' + user_session_id
    file_path = base_data_folder_path / user_id / session_folder_name / file_name
    df = pd.read_csv(file_path, names=colume_names, index_col=False)
    return df
def get_user_session_data(user_id: str, user_session_id: str) -> DataFrame: 
    """
    Combine accelerometer, gyroscope, and activity labels for a specific session of a user
    Note: Timestamps are ignored when joining accelerometer and gyroscope data.  
    :param user_id: user id
    :param user_session_id: user session id
    :return: combined DataFrame for a session
    """
    file_name = 'Activity.csv'
    colume_names = file_name_to_colume_names[file_name]
    activity_df = read_file(user_id=user_id, user_session_id=user_session_id, file_name=file_name, colume_names=colume_names)
    
    file_name = 'Accelerometer.csv'
    colume_names = file_name_to_colume_names[file_name]
    accel_df = read_file(user_id=user_id, user_session_id=user_session_id, file_name=file_name, colume_names=colume_names)
    
    file_name = 'Gyroscope.csv'
    colume_names = file_name_to_colume_names[file_name]
    gyro_df = read_file(user_id=user_id, user_session_id=user_session_id, file_name=file_name, colume_names=colume_names)
    
    # join accelerometer and gyroscope data ignoring timestamps
    measurements_df = accel_df.join(gyro_df, lsuffix='_accel', rsuffix='_gyro')
    full_df = measurements_df.join(activity_df.set_index('ID'), on='ActivityID' + '_accel')
    full_df = full_df.dropna()
    
    return full_df
def multiV_curvature(nbddata: DataFrame) -> float:  
    """
    Calculate multi V curvature
    :param nbddata: neighborhood of time t_i containing (t, x(t), y(t), z(t)), 
    where x(t), y(t), z(t) are the 3 out of the 6 features. 
    :return: multi V curvature
    """
    alpha = nbddata[["X_accel", "Y_accel","Z_accel"]]
    t = nbddata[["Systime_accel"]]
    t=t-t.iloc[0]
    # to fit each component of alpha by a cubic curve
    coef=[]
    for i in range(3):
        coef+=[np.polyfit(t.iloc[:,0], alpha.iloc[:,i], 2)]
    coefm=np.stack(coef) 
    v_0=coefm[:,2]
    v_1=coefm[:,1]
    v_2=coefm[:,0]
    curvature=2*np.linalg.norm(np.cross(v_1, v_2))/np.linalg.norm(v_1)**(3/2)
    return curvature
def multiV_torsion(nbddata: DataFrame) -> float:  
    """
    Calculate multi V torsion
    :param nbddata: neighborhood of time t_i containing (t, x(t), y(t), z(t)), 
    where x(t), y(t), z(t) are the 3 out of the 6 features. 
    :return: multi V torsion
    """
    alpha = nbddata[["X_accel", "Y_accel","Z_accel"]]
    t = nbddata[["Systime_accel"]]
    t=t-t.iloc[0]
    # to fit each component of alpha by a cubic curve
    coef=[]
    for i in range(3):
        coef+=[np.polyfit(t.iloc[:,0], alpha.iloc[:,i], 3)]
    coefm=np.stack(coef) 
    v_0=coefm[:,3]
    v_1=coefm[:,2]
    v_2=coefm[:,1]
    v_3=coefm[:,0]
    torsion=3*np.linalg.det(coefm[:,:3])/np.linalg.norm(np.cross(v_1, v_2))
    return torsion


# Get all user ids
user_ids = get_user_ids()
user_id = user_ids[0]
print(user_id, user_ids)
# Get all session ids for a specific user
user_session_ids = get_user_session_ids(user_id)
user_session_id = user_session_ids[0]
print(user_session_id, user_session_ids)
# Read one of the csv files for a user
file_name='Gyroscope.csv'
colume_names = file_name_to_colume_names[file_name]
read_file(user_id=user_id, user_session_id=user_session_id, file_name=file_name, colume_names=colume_names)
# Combine accelerometer, gyroscope, and activity labels for a specific session of a user
df = get_user_session_data(user_id=user_id, user_session_id=user_session_id)
# Accelerometer for a specific session of a user with activityId: 100669131000001 and plot
dff=df[df.ActivityID_accel == 100669131000001]
df_Ex=dff[["Systime_accel","X_accel", "Y_accel","Z_accel"]]
df_Ex[["X_accel", "Y_accel","Z_accel"]].plot()
plt.show()
# Multi-V time series curvature and plot
cur=[]
for i in range(0,df_Ex.shape[0],10):
    cur+=[multiV_curvature(df_Ex.iloc[i:i+10])]
plt.plot(cur,'.')   
plt.title('Multi-V time series curvature')
plt.show()
# Multi-V time series torsion and plot
cur=[]
for i in range(0,df_Ex.shape[0],10):
    cur+=[multiV_torsion(df_Ex.iloc[i:i+10])]
plt.plot(cur,'.')   
plt.title('Multi-V time series torsion')
plt.show()






