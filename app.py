# Import all of the dependencies
import streamlit as st
import os 
import cv2
import tensorflow as tf 
import numpy as np
from utils import load_data, num_to_char, load_video, load_alignments
from modelutil import load_lipnet_model
import base64
import ffmpeg
import tempfile

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('data','data', 's1'))

selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 
    # Rendering the video 
    with col1: 
        st.info('The video below displays the video with audio')
        file_path = os.path.join('data','data','s1', selected_video)
        
        try:
            # Create a temporary file for the converted video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Convert video to MP4 format
            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(stream, temp_path, vcodec='libx264', acodec='aac')
            ffmpeg.run(stream, overwrite_output=True)
            
            # Read the converted video file
            with open(temp_path, 'rb') as file:
                video_bytes = file.read()
            
            # Convert video to base64
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            # Create HTML video player
            video_html = f"""
            <video width="100%" controls>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
            
            # Display the video player
            st.components.v1.html(video_html, height=400)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"Error displaying video: {str(e)}")

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        
        # Load and process the video frames
        try:
            frames = load_video(file_path)
            st.info('This is the output of the machine learning model as tokens')
            model = load_lipnet_model()
            
            if model is None:
                st.error("Failed to load the model. Please check if the model file exists.")
            else:
                # Get prediction
                yhat = model.predict(tf.expand_dims(frames, axis=0))
                decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
                
                # Filter out padding tokens (-1)
                valid_tokens = decoder[decoder != -1]
                
                # Display raw tokens
                st.text("Raw tokens (excluding padding):")
                st.text(valid_tokens)
                
                # Convert prediction to text
                st.info('Decode the raw tokens into words')
                converted_prediction = tf.strings.reduce_join(num_to_char(valid_tokens)).numpy().decode('utf-8')
                st.text("Predicted text:")
                st.text(converted_prediction)
                
                # Load and display ground truth from alignment file
                try:
                    file_name = os.path.splitext(selected_video)[0]
                    alignment_path = os.path.join('data', 'data', 'alignments', 's1', f'{file_name}.align')
                    ground_truth = load_alignments(alignment_path)
                    ground_truth_text = tf.strings.reduce_join(num_to_char(ground_truth)).numpy().decode('utf-8')
                    st.info('Ground truth text:')
                    st.text(ground_truth_text)
                except Exception as e:
                    st.warning(f"Could not load alignment file: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        