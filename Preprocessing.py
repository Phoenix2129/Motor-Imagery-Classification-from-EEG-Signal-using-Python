!pip install mne
import mne
import pandas as pd

# Load the GDF file
file_path = "/content/drive/MyDrive/A01T.gdf"
raw_data = mne.io.read_raw_gdf(file_path, preload=True)

# Apply a bandpass filter between 10 and 30 Hz (for EEG data)
raw_data.filter(l_freq=10, h_freq=30)

# Convert EEG data to a DataFrame
eeg_df = raw_data.to_data_frame()

# Extract events and event IDs from annotations
events, event_id = mne.events_from_annotations(raw_data)

# Create an empty label column initialized with NaN
eeg_df['Label'] = pd.NA

# Assign labels based on events
for event in events:
    start_sample = event[0]  # Start sample of the event
    event_type = event[2]  # Event type ID

    # Map specific event types to motor imagery classes (1, 2, 3, 4)
    if event_type in [7, 8, 9, 10]:  
        class_label = event_type - 6  
        eeg_df.at[start_sample, 'Label'] = class_label

eeg_df['Label'] = eeg_df['Label'].ffill()

# Save the EEG data and labels to a CSV file
output_csv_file = '/content/EEG_with_labels.csv'
eeg_df.to_csv(output_csv_file, index=False)

print("EEG data with labels saved to:", output_csv_file)

csv_file_path = '/content/EEG_with_labels.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Print the DataFrame
print(df)

file_path = "/content/EEG_with_labels.csv"
df = pd.read_csv(file_path)
columns_to_delete = ["EOG-left", "EOG-right","EOG-central"]  # Replace with your column names

# Delete extra channels
df.drop(columns=columns_to_delete, inplace=True)
output_file_path = "modified.csv"
df.to_csv(output_file_path, index=False)
r = pd.read_csv("modified.csv")
print(r)

input_file = 'modified.csv'
df = pd.read_csv(input_file)

# Specify the column name that contains the class labels
class_column = 'Label'

# Separate the data based on the class labels and save to separate CSV files
for label in df[class_column].unique():
    if pd.isna(label):  # Check for NaN class
        filtered_df = df[df[class_column].isna()]
        output_file = 'class_NaN.csv'
    else:
        filtered_df = df[df[class_column] == label]
        output_file = f'class_{label}.csv'

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    print(f'Saved {output_file}')


