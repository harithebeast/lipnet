import tensorflow as tf
from typing import List
import cv2
import os 
import numpy as np


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame = frame[190:236, 80:220]  # Crop frame
        frame = frame / 255.0  # Normalize pixel values
        frames.append(frame)

    cap.release()
    
    # Ensure correct shape (frames, height, width, channels)
    frames = np.array(frames, dtype=np.float32)
    frames = np.expand_dims(frames, axis=-1)  # Add channel dimension

    mean = np.mean(frames)
    std = np.std(frames) if np.std(frames) != 0 else 1  # Avoid division by zero
    
    return tf.convert_to_tensor((frames - mean) / std, dtype=tf.float32)

    
def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments