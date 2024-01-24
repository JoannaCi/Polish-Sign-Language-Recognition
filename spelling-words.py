# %%
import os
import json
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from IPython.display import display
import pandas as pd

from tqdm.auto import tqdm

# %%
# Create a list of lowercase letters from the filenames in the specified directory
letters = [l.split('.')[0].lower() for l in os.listdir("/home/ant/projects/psl/dataset/Videos/alphabet")]

# Define the vocabulary as a list containing '<start>' and '<eos>' tokens, along with the letters
vocabulary = ['<pad>', '<start>', '<eos>'] + letters

# Create a dictionary mapping each vocabulary item to its corresponding index
# Indexing starts from 1, so '<start>' is assigned index 1, '<eos>' is assigned index 2, and so on
vocabulary = {l: i for i, l in enumerate(vocabulary)}

# Display the resulting vocabulary dictionary
vocabulary

# %%
# Function to extract hand landmarks from a video
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
def landmarks_timeseries(video_path):
    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the frame rate for extracting landmarks
    frame_rate = 0.1

    # Calculate the number of frames to skip based on the frame rate
    frames_to_skip = int(fps * frame_rate)

    landmarks_data = []
    current_frame = 0

    # Loop through the frames of the video
    while cap.isOpened():
        # Set the position to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # Read the current frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks using Mediapipe
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0].landmark

            # Append the 3D coordinates of hand landmarks to the list
            landmarks_data.append([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks])

        # Move to the next frame based on the frames to skip
        current_frame += frames_to_skip

    # Release the video capture object
    cap.release()
    del cap

    # Reshape the landmarks data into a 2D array
    landmarks_data = np.array(landmarks_data).reshape(len(landmarks_data), -1)

    return landmarks_data

# %%
videos_path = "/home/ant/projects/psl/dataset/Videos/alphabet"
labels = []
landmarks = []
# Iterate through each file in the dynamic alphabet directory
for i, filename in enumerate(tqdm(os.listdir(videos_path))):
    if filename.endswith('.mp4'):
        video_path = os.path.join(videos_path, filename)

        label = filename.split('.')[0].lower()
        label = ['<start>', label, '<eos>']

        # Convert labels to their corresponding vocabulary indices
        label = [vocabulary[l] for l in label]
        labels.append(label)

        # Call the function 'landmarks_timeseries' to get landmarks from the video
        land = landmarks_timeseries(video_path)

        landmarks.append(land)

# %%
len(landmarks), len(labels)

# %%
videos_path = "/home/ant/projects/psl/dataset/Videos/words"
labels_words = []
landmarks_words = []
# Iterate through each file in the words directory
for i, filename in enumerate(tqdm(os.listdir(videos_path))):
    if filename.endswith('.mp4'):
        video_path = os.path.join(videos_path, filename)

        # Extract labels from the filename, including '<start>' and '<eos>' tokens
        label = ['<start>'] + list(filename.split('.')[0].lower()) + ['<eos>']

        # Convert labels to their corresponding vocabulary indices
        label = [vocabulary[l] for l in label]

        labels_words.append(label)

        # Call the function 'landmarks_timeseries' to get landmarks from the video
        land = landmarks_timeseries(video_path)
        landmarks_words.append(land)

# %%
# Define a mapping to fix certain characters in the labels
fix = {
    'Ć': 'ć',
    'Ę': 'ę',
    'Ł': 'ł',
    'Ń': 'ń',
    'Ó': 'O',
    'Ś': 'ś',
    'Ź': 'ź',
    'Ż': 'ż',
}

# Function to preprocess data from JSON files in the folder
def preprocess_data(labels_folder):
    labeled_with_landmarks_count = 0
    labeled_without_landmarks_count = 0
    data_rows = []  # List to store data rows
    labels = []  # List to store labels

    # Loop through JSON files in the folder
    for filename in tqdm(os.listdir(labels_folder)):
        if filename.endswith('.json'):
            with open(os.path.join(labels_folder, filename), 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                if 'hand_landmarks' in data:
                    labeled_with_landmarks_count += 1

                    # Extract landmarks data and flatten it into a list
                    landmarks_data = data['hand_landmarks']
                    row = []
                    for landmark_key in landmarks_data:
                        landmark = landmarks_data[landmark_key]
                        row.extend([landmark['x'], landmark['y'], landmark['z']])

                    # Extract and preprocess the label
                    l = data['label']
                    if l in fix:
                        l = fix[l]
                    label = ['<start>', l.lower(), '<eos>']
                    label = [vocabulary[l] for l in label]
                    data_rows.append(row)
                    labels.append(label)

                else:
                    labeled_without_landmarks_count += 1

    return data_rows, labels

labels_folder = "/home/ant/projects/psl/dataset/labels"
preprocessed_data, preprocessed_labels = preprocess_data(labels_folder)

# %% [markdown]
# _______________

# %% [markdown]
# ASIOWE TESTY
# 

# %%
# import pandas as pd

# # Load static labels from CSV file
# csv_file_path = '/home/ant/projects/psl/Polish-Sign-Language-Recognition/train_data.csv'
# static_labels_df = pd.read_csv(csv_file_path, header=None, names=['filename', 'label'], skiprows=1)



# %%
# csv_file_path_static = '/home/ant/projects/psl/Polish-Sign-Language-Recognition/train_data.csv'
# static_labels_df = pd.read_csv(csv_file_path_static, usecols=[0], header=None, names=['filename'], skiprows=1)
# data_static_new = []

# %%
def load_and_process_image(file_path):
    # Load the image in binary mode
    with open(file_path, 'rb') as file:
        image = Image.open(file)
        # You can add additional image processing logic here if needed
        processed_content = np.array(image)  # Convert the image to a NumPy array
    return processed_content

# %%
import tensorflow as tf

# %%

def read_static(landmarks_directory, filenames_df):
    ###LABELS
    # Iterate through each row in the static labels dataframe
    labels_static = []
    for i, row in filenames_df.iterrows():
        # Extract filename and label from the CSV file
        filename = row['filename']
        label = row['label']
        
        # Convert labels to their corresponding vocabulary indices
        label = ['<start>'] + list(label.lower()) + ['<eos>']
        label = [vocabulary[l] for l in label]
        labels_static.append(label)
    
    ### LANDMARKS
    # Iterate through each row in the filenames dataframe
    landmarks_static = []
    for i, row in filenames_df.iterrows():
        # Extract filename from the CSV file
        filename = row['filename']
        
        # Construct the full path to the file with landmarks
        landmarks_file_path = os.path.join(landmarks_directory, filename + '.json')
        
        # Check if the file exists before attempting to read landmarks
        if os.path.exists(landmarks_file_path):
            # Read landmarks from the file
            with open(landmarks_file_path, 'r') as landmarks_file:
                landmark_data = json.load(landmarks_file)
                
                # Extract relevant information from the JSON structure
                hand_landmarks = landmark_data.get("hand_landmarks", {})
                
                # Create a flat list of numerical values representing each hand landmark
                landmark_values = []
                for key, values in hand_landmarks.items():
                    if key.startswith("hand_landmark_"):
                        # Extract x, y, and z values directly
                        x, y, z = values.get("x", 0.0), values.get("y", 0.0), values.get("z", 0.0)
                        landmark_values.extend([x, y, z])
                
                # Debugging information
                print(f"Length of landmark_values for {filename}: {len(landmark_values)}")
                
                # Ensure the length of the list is exactly 63
                if len(landmark_values) != 63:
                    # Print the length of the list for debugging
                    print(f"Unexpected length of landmark_values for {filename}: {len(landmark_values)}")
                    
                    # Handle the unexpected length (optional)
                    # For example, you might choose to skip this sample
                    continue
                
                # Append the flat list of values to landmarks_static
                landmarks_static.append(landmark_values)
        else:
            # Handle the case when the file does not exist
            print(f"Landmarks file not found for {filename}")
    return landmarks_static, labels_static

# Print the extracted landmarks for verification
# for landmark_values in landmarks_static:
#     print(landmark_values)

# %%
# Load filenames from CSV file
csv_file_path = '/home/ant/projects/psl/Polish-Sign-Language-Recognition/train_data.csv'
filenames_df = pd.read_csv(csv_file_path, header=None, names=['filename', 'label'], skiprows=1)

# Specify the directory containing the files with landmarks
landmarks_directory = '/home/ant/projects/psl/dataset/labels'

train_landmarks_static, train_labels_static = read_static(landmarks_directory, filenames_df)

# Load filenames from CSV file
csv_file_path = '/home/ant/projects/psl/Polish-Sign-Language-Recognition/test_data.csv'
filenames_df = pd.read_csv(csv_file_path, header=None, names=['filename', 'label'], skiprows=1)

test_landmarks_static, test_labels_static = read_static(landmarks_directory, filenames_df)

# %%
len(train_landmarks_static), len(train_labels_static), len(test_landmarks_static), len(test_labels_static)

# %%
def reshape_landmarks_static(data):
    data = np.array(data)
    # Create a new list to store the modified static data
    data_static_new = []

    # Iterate through each element in the original static data
    for d in data:
        # Repeat the current element along a new axis a random number of times (between 2 and 6)
        d = np.repeat(d.reshape(1, -1), repeats=np.random.randint(2, 7), axis=0)
        data_static_new.append(d)
    return data_static_new

# %%
train_landmarks_static = reshape_landmarks_static(train_landmarks_static)
test_landmarks_static = reshape_landmarks_static(test_landmarks_static)

# %%
from sklearn.model_selection import train_test_split
train_landmarks_words, test_landmarks_words, train_labels_words, test_labels_words = train_test_split(landmarks_words, labels_words, test_size=0.2, random_state=42)

# %%
len(train_landmarks_words), len(train_labels_words), len(test_landmarks_words), len(test_labels_words)


# %%
# Litery dynamiczne, slowa, litery statyczne
data = landmarks + train_landmarks_words + train_landmarks_static
all_labels = labels + train_labels_words + train_labels_static


# %%
# data = landmarks + train_landmarks_words * 5 + train_landmarks_static
# all_labels = labels + train_labels_words * 5 + train_labels_static

# # %%
# data = train_landmarks_words
# all_labels = train_labels_words

# %%
# slowa, litery statyczne
test_data = test_landmarks_words + test_landmarks_static
test_all_labels = test_labels_words + test_labels_static

# %%
len(data), len(all_labels), len(test_data), len(test_all_labels)

# %% [markdown]
# __________________ 

# %%
# Ensure that the number of samples is consistent between data and all_labels
assert len(data) == len(all_labels), "Number of samples in data and all_labels must be the same."


# %%
def masked_loss(y_true, y_pred):
    # Initialize SparseCategoricalCrossentropy loss with 'from_logits' and 'reduction' parameters
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    # Calculate the cross-entropy loss for each item in the batch
    loss = loss_function(y_true, y_pred)

    # Create a binary mask to filter out padding elements (where y_true is 0)
    mask = tf.cast(y_true != 0, tf.float32)

    # Apply the mask to the calculated losses
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

# %%
def accuracy_for_letters(y_true, y_pred):
    result = tf.cast(tf.cast(y_true, tf.int64) == tf.argmax(y_pred, axis=-1), tf.float32)
    # print(result)

    # Create a binary mask to filter out padding elements (where y_true is 0)
    mask = tf.math.logical_and(y_true != 0, y_true != 1)
    mask = tf.math.logical_and(mask, y_true != 2)
    mask = tf.cast(mask, tf.float32)
    # print(mask)

    # Apply the mask to the calculated losses
    result *= mask
    # print(result)
    s = tf.reduce_sum(result, axis=-1)
    # print(s)
    m = tf.reduce_sum(mask, axis=-1)
    # print(m)
    r = tf.cast(s, tf.int64) ==  tf.cast(m, tf.int64)
    r = tf.cast(r, tf.float32)
    # print(r)
    # return tf.reduce_sum(result) / tf.reduce_sum(mask)
    return tf.reduce_mean(r)

# %%
# accuracy_for_letters(padded_test_words_Labels[:3], model.predict(padded_test_words[:3]))

# %%
max_len_input = 0
for d in data:
    max_len_input = max(max_len_input, len(d))
for d in test_data:
    max_len_input = max(max_len_input, len(d))
max_len_input

# %%
max_len_output = 0
for d in all_labels:
    max_len_output = max(max_len_output, len(d))
for d in test_all_labels:
    max_len_output = max(max_len_output, len(d))
max_len_output

# %%
len(data), len(all_labels), len(test_data), len(test_all_labels)

# %%
# Pad input sequences (data) with zeros using "post" padding

padded_train_inputs = tf.keras.utils.pad_sequences(data, maxlen=max_len_input, dtype="float32", padding="post")
padded_train_outputs = tf.keras.utils.pad_sequences(all_labels, maxlen=max_len_output, dtype="int32", padding="post")

# padded_inputs.shape, padded_outputs.shape

# %%
# Pad output sequences (train_labels and test_labels) with zeros using "post" padding

padded_test_inputs = tf.keras.utils.pad_sequences(test_data, maxlen=max_len_input, dtype="float32", padding="post")
padded_test_outputs = tf.keras.utils.pad_sequences(test_all_labels, maxlen=max_len_output, dtype="float32", padding="post")

# %%
padded_train_inputs.shape, padded_train_outputs.shape, padded_test_inputs.shape, padded_test_outputs.shape

# %%
from tensorflow import keras


# Define the input dimension, vocabulary size, and create a Sequential model
input_dim = 63
vocab_size = len(vocabulary)

model = keras.Sequential(
    [
        # Input layer with shape (sequence_length, input_dim)
        keras.Input(shape=(padded_train_inputs.shape[1], input_dim), dtype="float32"),

        # Masking layer to handle variable-length sequences
        keras.layers.Masking(),

        # LSTM layer with 32 units, returning a single output for each sequence
        # keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=False),

        # Repeat the output vector for each time step in the output sequence
        keras.layers.RepeatVector(padded_train_outputs.shape[1]),

        # LSTM layer with 64 units, returning a sequence of vectors
        keras.layers.LSTM(32, return_sequences=True),

        # TimeDistributed layer to apply Dense layer to each time step independently
        keras.layers.TimeDistributed(keras.layers.Dense(vocab_size)),
    ]
)

# Display the model summary
model.summary()

# Compile the model using the custom masked loss function and Adam optimizer
model.compile(
    loss=masked_loss,
    optimizer=tf.keras.optimizers.legacy.Adam(1e-3),
    metrics=[accuracy_for_letters]
)

# %%
model.fit(padded_train_inputs, padded_train_outputs, validation_split=0.2, epochs=200)
model.save('script_model.h5')

# %%
model.evaluate(padded_test_inputs, padded_test_outputs)

# %%
padded_test_words = tf.keras.utils.pad_sequences(test_landmarks_words, maxlen=max_len_input, dtype="float32", padding="post")
padded_test_words_Labels = tf.keras.utils.pad_sequences(test_labels_words, maxlen=max_len_output, dtype="float32", padding="post")

# %%
model.evaluate(padded_test_words, padded_test_words_Labels)

# %%
padded_test_static = tf.keras.utils.pad_sequences(test_landmarks_static, maxlen=max_len_input, dtype="float32", padding="post")
padded_test_static_Labels = tf.keras.utils.pad_sequences(test_labels_static, maxlen=max_len_output, dtype="float32", padding="post")

# %%
model.evaluate(padded_test_static, padded_test_static_Labels)

# %%
vocabulary
inv_vocab = {v: k for k, v in vocabulary.items()}

# %%
def translate_word(word):
    new_word = ''
    for c in word:
        new_c = inv_vocab[c]
        if new_c == '<start>':
            continue
        elif new_c == '<eos>' or new_c == '<pad>':
            break
        else:
            new_word += new_c
    return new_word


# %%
def print_words(pred, labels):
    pred = tf.argmax(pred, axis=-1).numpy()
    for p, t in zip(pred, labels):
        p = translate_word(p)
        t = translate_word(t)
        print(p, t)

# %%
print_words(model.predict(padded_test_static), padded_test_static_Labels)


# %%
print_words(model.predict(padded_test_words), padded_test_words_Labels)


