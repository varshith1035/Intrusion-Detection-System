import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from intrusion_detector import IntrusionDetector
from zone_detector import ZoneDetector
from utils import validate_coordinates, create_download_link
def main():
    st.set_page_config(
        page_title="Intrusion Detection System",
        page_icon="🚨",
        layout="wide"
    )
    
    st.title("🚨 Offline Intrusion Detection System")
    st.markdown("Upload a video file and define restricted zones for person detection using motion detection")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        # Zone type selection
        zone_type = st.selectbox(
            "Zone Type",
            ["Line", "Polygon"],
            help="Select the type of restricted zone"
        )
        
        # Detection sensitivity
        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence for person detection"
        )
        
        # Zone coordinates input
        st.subheader("Zone Coordinates")
        
        if zone_type == "Line":
            st.markdown("**Define a line (2 points):**")
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input("X1", value=100, min_value=0)
                y1 = st.number_input("Y1", value=200, min_value=0)
            with col2:
                x2 = st.number_input("X2", value=500, min_value=0)
                y2 = st.number_input("Y2", value=200, min_value=0)
            
            zone_coords = [(x1, y1), (x2, y2)]
            
        else:  # Polygon
            st.markdown("**Define polygon points (minimum 3):**")
            num_points = st.number_input("Number of Points", min_value=3, max_value=10, value=4)
            
            zone_coords = []
            for i in range(num_points):
                col1, col2 = st.columns(2)
                with col1:
                    x = st.number_input(f"X{i+1}", value=100 + i*50, min_value=0, key=f"x_{i}")
                with col2:
                    y = st.number_input(f"Y{i+1}", value=100 + (i%2)*100, min_value=0, key=f"y_{i}")
                zone_coords.append((x, y))
    
    # Main content area
    if uploaded_file is not None:
        # Create temporary file for uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Display video info
        cap = cv2.VideoCapture(tfile.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps
        cap.release()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("Resolution", f"{width}x{height}")
        with col3:
            st.metric("FPS", fps)
        
        # Validate coordinates
        valid_coords, error_msg = validate_coordinates(zone_coords, width, height)
        
        if not valid_coords:
            st.error(f"Invalid coordinates: {error_msg}")
        else:
            st.success("Zone coordinates are valid!")
            
            # Display zone preview
            st.subheader("Zone Preview")
            preview_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            zone_detector = ZoneDetector(zone_coords, zone_type.lower())
            zone_detector.draw_zone(preview_frame)
            
            st.image(preview_frame, caption="Restricted Zone Preview", use_column_width=True)
            
            # Process video button
            if st.button("🎯 Process Video", type="primary"):
                process_video(tfile.name, zone_coords, zone_type.lower(), confidence_threshold)
        
        # Cleanup
        os.unlink(tfile.name)
    else:
        st.info("👆 Please upload a video file to get started")
        
        # Show example coordinates
        st.subheader("Example Coordinates")
        st.markdown("""
        **Line Example (Horizontal barrier):**
        - Point 1: (100, 300)
        - Point 2: (500, 300)
        
        **Polygon Example (Rectangular area):**
        - Point 1: (200, 200)
        - Point 2: (400, 200)
        - Point 3: (400, 350)
        - Point 4: (200, 350)
        """)
def process_video(video_path, zone_coords, zone_type, confidence_threshold):
    """Process video with intrusion detection"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize intrusion detector
        detector = IntrusionDetector(
            zone_coords=zone_coords,
            zone_type=zone_type,
            confidence_threshold=confidence_threshold
        )
        
        status_text.text("Initializing motion detection model...")
        
        # Create output filename
        output_path = "processed_video.mp4"
        
        # Process video
        intrusion_count = detector.process_video(
            input_path=video_path,
            output_path=output_path,
            progress_callback=lambda p: progress_bar.progress(p)
        )
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Show results
        st.success(f"✅ Video processed successfully!")
        st.info(f"🚨 Total intrusions detected: {intrusion_count}")
        
        # Provide download link
        if os.path.exists(output_path):
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=file.read(),
                    file_name="intrusion_detection_output.mp4",
                    mime="video/mp4"
                )
            
            # Display first frame with detections as preview
            cap = cv2.VideoCapture(output_path)
            ret, frame = cap.read()
            if ret:
                st.subheader("Preview (First Frame)")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Processed Video Preview", use_column_width=True)
            cap.release()
            
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        progress_bar.empty()
        status_text.empty()
if __name__ == "__main__":
    main()
