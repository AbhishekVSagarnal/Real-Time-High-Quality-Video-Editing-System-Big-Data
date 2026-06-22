import subprocess
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename='chunking.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_video_info(video_path):
    try:
        # Get video information using ffmpeg
        cmd = ['ffmpeg', '-i', video_path]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Extract FPS
        for line in output.split('\n'):
            if 'fps' in line:
                fps = float(line.split(' ')[-2])  # Extract FPS
                logging.info(f"FPS: {fps}")
                return fps
    except subprocess.CalledProcessError as e:
        logging.error("Error retrieving video info: " + str(e))
    return None

def chunk_video(video_path, num_chunks):
    fps = get_video_info(video_path)
    if fps is None:
        return
    
    # Get video length in seconds
    video_length = float(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries',
         'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]).strip())
    
    # Calculate total frames
    total_frames = int(fps * video_length)
    logging.info(f"Total frames: {total_frames}")

    # Calculate chunk duration
    chunk_duration = video_length / num_chunks
    logging.info(f"Chunk duration: {chunk_duration}")

    # Create chunks
    for i in range(num_chunks):
        start_time = i * chunk_duration
        output_filename = f"{os.path.splitext(video_path)[0]}_keyframe_chunk_{i+1}.mp4"
        cmd = ['ffmpeg', '-i', video_path, '-ss', str(start_time), '-t', str(chunk_duration),
               '-vf', 'select=eq(pict_type\\,I)', '-vsync', 'vfr', output_filename]
        
        start_time_logging = logging.info(f"Creating chunk {i+1} at {start_time:.2f}s")
        subprocess.run(cmd, stderr=subprocess.PIPE)
        logging.info(f"Chunk created: {output_filename}")

if __name__ == "__main__":
    video_file = "input_video.mp4"  # Change this to your video file
    for num_chunks in range(2, 11):  # Try varying chunk numbers
        chunk_video(video_file, num_chunks)
