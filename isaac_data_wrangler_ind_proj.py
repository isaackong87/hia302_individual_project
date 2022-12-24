#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###---------------
# Individual Project HIA302 
# Isaac Kong
# Data Collection and Preparation 
###----------------


# In[1]:


import pandas as pd
import os, sys
import numpy as np
import csv


# In[2]:


data_file = 'horse-colic.txt'
data_names = 'horse-colic.names.txt'


# In[3]:


### Method 1

### Step 1 convert txt files to csv files
### Here are two different ways to convert a text file to a CSV file in Python:
### 1.0 Using the csv module:


# # Open the input file in read mode (for horse-colic.txt file)
with open(data_file, 'r') as input_file:
  # Open the output file in write mode
  with open('horse-colic_method_1.csv', 'w', newline='') as output_file:
    # Create a CSV writer object
    writer = csv.writer(output_file)

    # Iterate over the lines in the input file
    for row in input_file:
      # Split the line into fields using the delimiter specified
      fields = row.split(',')
      # Write the fields to the output file
      writer.writerow(fields)


# In[5]:


### Method 2
lines_list = list()

with open(data_file, 'r') as fh:
    
    for line in fh:
        tmp_line = line.strip().split(',')
        lines_list.append(tmp_line)

len(lines_list)
df_tmp1 = pd.DataFrame(lines_list)

# Writing into a CSV
df_tmp1.to_csv('horse-colic_method_2.csv', index=False)

df_tmp1.head()


# In[6]:


### Method 3
### Step 1 convert txt files to csv files
### 2.0 Using the pandas library:


# Read the input file into a Pandas dataframe (for horse-colic.txt file)
df_tmp2 = pd.read_csv(data_file, sep=',', header=None, dtype = str)

# Write the dataframe to a CSV file
df_tmp2.to_csv('horse-colic_method_3.csv', index=False )

df_tmp2.head()


# In[7]:


### Below show how to read a CSV file into a Pandas dataframe in a Jupyter notebook in Python
### and display the first few rows of the dataframe in the Jupyter notebook.

# Read the input CSV file into a Pandas dataframe (method 2)
## Use argument dtype=str to make sure variables are taken as they are in the raw data file
df = pd.read_csv('horse-colic_method_2.csv', sep=',', dtype = str)

df.info()
df.head()


# In[8]:


# Read the input CSV file into a Pandas dataframe (method 3)
## Use argument dtype=str to make sure variables are taken as they are in the raw data file
df = pd.read_csv('horse-colic_method_3.csv', sep=',', dtype = str)

df.info()
df.head()


# In[9]:


### Replace all '?' with NaN 
df = df.replace('?', np.nan)
df


# In[10]:


# Writing df into a CSV after replacing '?' with 'NaN'
df.to_csv('horse-colic-with-NaN.csv', index=False)


# In[11]:


# Import the re (regular expression) module
import re
names_lines = list() # Create an empty list to store processed lines

# Open the file specified by data_names in read mode and store it in the variable fh (file handler)
with open(data_names, 'r') as fh:
    
    # Iterate over each line in the file
    for line in fh:
        # Use a regular expression to check if the line starts with a number and a colon
        if(re.match(r'^[0-9]+:', line.strip())):
            
            # If the line matches the pattern, split it on the colon to separate the number from the rest of the line
            tmp_line = line.strip().split(':')
            
            # Decrease the number by 1 and convert it back to a string
            tmp_line[0] = str(int(tmp_line[0]) - 1) 
            
            # Remove certain characters, strip leading and trailing whitespace, and capitalize the rest of the line
            tmp_line[1] = tmp_line[1].replace('?', '').replace('_', '').replace('-', '').replace('\'s', '').strip().capitalize()
            
            # Append the modified tmp_line to names_lines
            names_lines.append(tmp_line)
            
# Convert the list of processed lines to a dictionary        
names_dict = dict(names_lines)
# Print the final dictionary
names_dict


# In[12]:


# Below are step to This code is accessing the keys in the 
# dictionary names_dict and assigning it the value to it. 
# The key is used to retrieve the value associated with it in the dictionary,
# and the assignment operator = is used to assign a new value to the key.
names_dict['2'] = 'HospitalNo'
names_dict['3'] = 'RecTemp'
names_dict['5'] = 'RespRate'
names_dict['6'] = 'TempExtrem'
names_dict['7'] = 'PeriPulse'
names_dict['8'] = 'MucousMemb'
names_dict['9'] = 'CapRefTime'
names_dict['10'] = 'PainLevel'
names_dict['11'] = 'Peristals'
names_dict['12'] = 'AbdmDisten'
names_dict['13'] = 'NasgasTub'
names_dict['14'] = 'NasgasRe'
names_dict['15'] = 'NasgasRepH'
names_dict['16'] = 'RecExFeces'
names_dict['18'] = 'PacCellVol'
names_dict['19'] = 'TolProtein'
names_dict['20'] = 'AbdAppear'
names_dict['21'] = 'AbdTolPro'
names_dict['23'] = 'SurLesion'
names_dict['24'] = 'LesionT1'
names_dict['25'] = 'LesionT2'
names_dict['26'] = 'LesionT3'


# In[13]:


# Rename the columns of the DataFrame `df` using the dictionary `names_dict`
df_renamed = df.rename(columns=names_dict)

# Save the modified DataFrame to a CSV file named 'horse-colic-with-title.csv', with the index disabled
df_renamed.to_csv('horse-colic-with-title.csv', index=False)

# Print the modified DataFrame
df_renamed


# In[14]:



# Read a CSV file into a pandas dataframe and specify the data type of each column as str
df_new = pd.read_csv('horse-colic-with-title.csv', dtype=str)

# Display the dataframe
df_new


# In[15]:


# Count the number of missing (NaN) values in each column of `df_new`
NaN_counts = df_new.isna().sum()

# Print the counts
NaN_counts


# In[16]:


# Print the total number of missing values in all columns
NaN_counts.sum()


# In[17]:


# Create a horizontal bar chart of the number of missing (NaN) values in each column of `df_new`,
# with a blue color, a figure size of (15,15), and a title
plt_tmp = NaN_counts.plot(kind='barh', figsize=(15,15), title="Count of NaN in Columns", color="blue")

# Iterate through each bar in the bar chart
for container in plt_tmp.containers:
    # Add a label to each bar with the count of missing values
    plt_tmp.bar_label(container, fmt="%1d")


# In[18]:


import matplotlib.pyplot as plt

# Get the column names and number of missing values as separate lists
columns = list(NaN_counts.index)
values = list(NaN_counts.values)

# Create a figure with a specified figure size
fig, ax = plt.subplots(figsize=(10, 10))

# Create a horizontal bar chart using the column names as the x-axis labels
# and the number of missing values as the bar heights
ax.barh(columns, values, color='blue')

# Add a title to the chart
ax.set_title('Count of NaN in Columns')

# Iterate through the values and add a label to each bar with the count of missing values
for i, v in enumerate(values):
    ax.text(v + 0.5, i, str(v), color='black', fontweight='bold')

# Display the chart
plt.show()


# In[19]:


# Sort the NaN_counts Series by the number of missing values
sorted_counts = NaN_counts.sort_values(ascending=False)

# Get the column names and number of missing values as separate lists
columns = list(sorted_counts.index)
values = list(sorted_counts.values)

# Create a figure with a specified figure size
fig, ax = plt.subplots(figsize=(10, 10))

# Create a horizontal bar chart using the column names as the x-axis labels
# and the number of missing values as the bar heights
ax.barh(columns, values, color='blue')

# Add a title to the chart
ax.set_title('Count of NaN in Columns (Sorted)')

# Iterate through the values and add a label to each bar with the count of missing values
for i, v in enumerate(values):
    ax.text(v + 0.5, i, str(v), color='black', fontweight='bold')

# Display the chart
plt.show()


# In[20]:


# retrieve data type
# Get information about the dataframe, including the data type of each column
df_new.info()


# In[21]:


# retrieve data type
# Get the data type of each column in the dataframe
df_new.dtypes


# In[22]:


### IF you drop rows with NaNs, you'll end up with only 6 rows ==> DO NOT ADVISE TO DROP 
df_new.dropna()


# In[23]:


# Convert all columns of the dataframe to float data type
df_new.astype(float)


# In[25]:


from sklearn.impute import SimpleImputer

# Load the dataset that's already processed and clean
df_new = pd.read_csv('horse-colic-with-title.csv', dtype=str)

# Convert datatypes object (str) to float in order to process them in Numpy, use attribute .values to convert dataframe into Numpy array
data = df_new.astype(float).values

# Get number of columns in numpy matrix array  
totalCol = data.shape[1]
# Get indecies of every column in array except for column at index 23 (Surgical Lesion)
ix = [i for i in range(totalCol) if i != 23]
# X is training dataset (numpy array)
X = data[:, ix]
y = data[:, 23]

# Print sum of all missing values in array dataset, use flatten to convert 2D array into 1D array
print("Missing %d" % sum(np.isnan(X).flatten()))

### Initiate model Simple Imputer
imputer = SimpleImputer(strategy='mean')
## Fit regression model on the training set X 
imputer.fit(X)
### Apply the imputer model on dataset X to impute missing values
Xtrans = imputer.transform(X)

# Print percentage of missing data
print("Missing %d" % sum(np.isnan(Xtrans).flatten()))

