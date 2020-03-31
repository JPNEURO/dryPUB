'''

1) Data Loading.
    - Assumes that the data has been arranged into Samples x Channels x Events x Trials.
    - Assumes the data array is comprised exclusively of electrode data, no event triggers or acqusition timing information.
    - Assumes time-series data are stored in a separate matric Time-Stamps x Events x Trials.
    - Assumes that the labels are stored in a separate variable in numeric format.

2) Data Segmentation.
3) Collection assessment.

# The principle aim is to focus on getting the data down to a single event as quickly as possible.

'''

import os
import numpy as np


def get_files(direc, keyword, ext, full_loc):
    '''
    Function grabs all filenames in data folder for rapid loading.

    Inputs:

    direc           = location of data folder.
    keyword         = if the data folder is cluttered with non-data files which share the same file extension
                      file extension identify a set of characters unique to the data files to parse accurately.
                      Having data files labelled with a fixed leading trail of zeroes makes this much simpler.
    extension       = the extension format of the data files for example: '.csv', '.npy', '.npz'.
    full_loc        = setting for switch between generating a list of ONLY file_names (FALSE) or full file
                      locations and file names appended.

    Outputs:

    file_names      = list of filename strings.

    Examples:

    file_names = get_files(direc='C:/P300_Project/Data/', keyword='000', ext='.npz', full_loc=True)
    file_names = get_files(direc='C:/P300_Project/Data', keyword=None, ext='.mat', full_loc=False)


    '''


    if keyword == None:
        eeg_files = [i for i in os.listdir(direc) if os.path.splitext(i)[1] == ext]
    elif keyword != None:
        file_temp = [i for i in os.listdir(direc) if os.path.splitext(i)[1] == ext]
        eeg_files = []
        for j in range(len(file_temp)):
            if file_temp[j].find(keyword) != -1:
                eeg_files.append(file_temp[j])

    if full_loc == True:
        _files = []
        for k in range(len(eeg_files)):
            # eeg_files[k] = np.append(direc, eeg_files[k])
            _files = np.append(_files, [direc + eeg_files[k]])
        eeg_files = _files

    return eeg_files


def folder_gen(data_folder):
    '''

    Creates folder for 'Converted_Data/' in the non-numpy format './Data/' folder.

    Inputs:

    data_folder = folder containing data to be converted.

    Ouputs:

    conv_folder = new converted data folder name.

    Example:

    folder_gen(data_folder='C:/P300_Project/Data/')

    '''

    conv_folder = data_folder + '/Converted_Data/'

    import os
    # Create Data Directory.
    if not os.path.exists(conv_folder):
        os.makedirs(conv_folder)
        print('Converted Data Directory Created.')
    else:
        print('Conveted Data Directory Already Exists.')
    return conv_folder


def del_cont(folder):
    'Deletes contents / files within folder specified.'
    import os
    import shutil

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def form_conv(file_names, data_folder):
    '''

    File conversion to numpy format, design for positioning inside for loop.

    1) Auto-detect file type '.mat' / '.csv', if .npy or .npz then print('Data already in correct numpy-format.')

    2) Create new direc + '/Converted_Data/' directory.

    3) Convert .mat or .csv files to numpy.

    4) Save converted files, storing new file names to return variable.

    '''

    # Imports
    import os
    import scipy.io as sio

    '---Detect File ID---'
    # Reference: https://www.tutorialspoint.com/How-to-extract-file-extension-using-Python
    ext = os.path.splitext(file_names[0])[1]
    print('\n', 'Original File Format: ', ext)

    if ext == '.npy' or ext == '.npz':
        print('Data already in correct numpy-format.')
    else:
        # Create create '/Converted_Data/' folder.
        conv_folder = folder_gen(data_folder)
        if ext == '.mat':
            '---Grab Electrode Header---'
            elecs = sio.loadmat(data_folder + '/Header.mat')
            elec_name = []
            n = sorted(elecs.keys())
            num_elecs = len(elecs[n[0]][0])
            for i in range(num_elecs):
                elec_name = np.append(elec_name, elecs[n[0]][0][i])
        elif ext == '.csv':
                import csv
                with open(data_folder + '/Header.csv', newline='') as f:
                    reader = csv.reader(f)
                    elec_name = list(reader)
                    elec_name = elec_name[0]
        print('Elec_Name: ', elec_name)
        # Iterate through non-numpy data files.
        for i in range(len(file_names)):
            print('---: Subject', i + 1)
            file_id = os.path.splitext(file_names[0])[0]
            if ext == '.mat':
                conv_dict = sio.loadmat(file_names[i])
                from operator import itemgetter
                conv_vals = conv_dict.values()
                conv_items = conv_dict.values()
                # Generate ordered and indexable dictionary key list.
                l = sorted(conv_dict.keys())
                conv_data = conv_dict[l[-1]]
                conv_sing = conv_data[:, -2]
            if ext == '.csv':
                import pandas as pd
                # Read the CSV into a pandas data frame, REF: https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array
                conv_data = pd.read_csv(file_names[i], delimiter=',')
                conv_data = conv_data.to_numpy()
            '---Data Info---'
            print('DATA DIMS: ',  conv_data.shape)
            '---Plot---'
            plotter = 0
            if plotter == 1:
                import matplotlib.pyplot as plt
                plt.plot(conv_data)
                plt.legend(labels=elec_name, loc='upper right', fontsize='xx-small', ncol=5) #
                plt.show()
            saver = 1
            if saver == 1:
                '---Save---'
                head, sub_id = os.path.split(file_names[i])
                sub_id = sub_id[0:-4]
                print('Conv Folder: ', conv_folder)
                print('Sub ID: ', sub_id)
                # print('FILE ID: ', conv_folder + '.npy')
                np.save((conv_folder + sub_id + '.npy'), conv_data)

'---File Location---'
style = 'mat'
data_folder = ('C:/P300_Project/BCI_Brain_Inv/Data/')
data_folder = ('C:/PhD_Publication/AlpData/')
data_folder = ('C:/PhD_Publication/Visual_ERP_BNCI/')
# REF: https://zenodo.org/record/3268762
data_folder = ('C:/PhD_Publication/ERP_ZEN_CON/')
if style == 'numpy':
    file_names = get_files(direc=data_folder, keyword=None, ext='.npz', full_loc=True)
if style == 'mat':
    file_names = get_files(direc=data_folder, keyword='group', ext='.mat', full_loc=True)
if style == 'csv':
    file_names = get_files(direc=data_folder, keyword='group', ext='.csv', full_loc=True)

print(file_names[0:5])

'---File Loading---'
form_conv(file_names, data_folder)
