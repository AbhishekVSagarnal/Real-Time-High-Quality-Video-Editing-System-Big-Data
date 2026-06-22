import subprocess
import os
import logging
import sys
from shutil import which
import time
import csv
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, filename='chunking.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def setup_output_directory():
    """Create output directory for the current run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "chunking_output"
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_terminal_output(output_text, run_dir, num_chunks):
    """Save terminal output to a text file"""
    output_file = os.path.join(run_dir, f"terminal_output_{num_chunks}_chunks.txt")
    with open(output_file, 'w') as f:
        f.write(output_text)

def create_csv_file(run_dir):
    """Create and initialize CSV file with headers"""
    csv_file = os.path.join(run_dir, "chunking_metrics.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Number of Chunks', 'Chunk Number', 'Chunk Length (s)', 'Processing Time (s)'])
    return csv_file

def append_to_csv(csv_file, num_chunks, chunk_num, chunk_length, processing_time):
    """Append a row of data to the CSV file"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([num_chunks, chunk_num, f"{chunk_length:.2f}", f"{processing_time:.2f}"])

def check_ffmpeg():
    """Check if FFmpeg is available in the system PATH"""
    print("Checking for FFmpeg installation...")
    if which('ffmpeg') is None or which('ffprobe') is None:
        logging.error("FFmpeg or FFprobe not found. Please install FFmpeg and add it to your PATH")
        print("Error: FFmpeg or FFprobe not found. Please install FFmpeg and add it to your PATH")
        sys.exit(1)
    print("✓ FFmpeg found successfully\n")

def get_video_info(video_path):
    print(f"Analyzing video: {os.path.basename(video_path)}")
    try:
        # Get video information using ffprobe
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        output = subprocess.check_output(cmd, universal_newlines=True).strip()
        
        # Parse the frame rate (comes in format num/den, e.g., "30000/1001")
        if '/' in output:
            num, den = map(float, output.split('/'))
            fps = num / den
        else:
            fps = float(output)
            
        logging.info(f"FPS: {fps}")
        print(f"✓ Video FPS: {fps:.2f}")
        return fps
        
    except subprocess.CalledProcessError as e:
        logging.error("Error retrieving video info: " + str(e))
        print("❌ Error retrieving video info")
        print(f"Error details: {str(e)}")
    except FileNotFoundError:
        logging.error("FFmpeg not found. Please install FFmpeg and add it to your PATH")
        print("❌ FFmpeg not found")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")
    return None

def chunk_video(video_path, num_chunks, run_dir, csv_file):
    terminal_output = []
    def log_print(message):
        print(message)
        terminal_output.append(message)

    log_print(f"\n{'='*50}")
    log_print(f"Starting chunking process for {num_chunks} chunks")
    log_print(f"{'='*50}")
    
    # Create subfolder for this chunk operation
    chunks_dir = os.path.join(run_dir, f"{num_chunks}_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    operation_start_time = time.time()
    fps = get_video_info(video_path)
    if fps is None:
        return
    
    log_print("\nCalculating video duration...")
    try:
        # Get video length in seconds
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        video_length = float(subprocess.check_output(cmd, universal_newlines=True).strip())
        
        # Calculate total frames
        total_frames = int(fps * video_length)
        log_print(f"✓ Total video duration: {video_length:.2f} seconds")
        log_print(f"✓ Total frames: {total_frames}")

        # Calculate chunk duration
        chunk_duration = video_length / num_chunks
        log_print(f"✓ Each chunk will be approximately {chunk_duration:.2f} seconds\n")
        
        total_chunk_time = 0
        
        # Create chunks
        for i in range(num_chunks):
            chunk_start_time = time.time()
            start_time = i * chunk_duration
            output_filename = os.path.join(chunks_dir, f"chunk_{i+1}.mp4")
            
            # Modified FFmpeg command for better reliability
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files without asking
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(chunk_duration),
                '-c:v', 'libx264',  # Use H.264 codec
                '-c:a', 'aac',      # Use AAC for audio
                '-force_key_frames', f'expr:gte(t,{start_time})',  # Force keyframe at start
                output_filename
            ]
            
            log_print(f"Processing chunk {i+1}/{num_chunks} [Starting at {start_time:.2f}s]...")
            result = subprocess.run(cmd, stderr=subprocess.PIPE, universal_newlines=True)
            
            chunk_processing_time = time.time() - chunk_start_time
            total_chunk_time += chunk_processing_time
            
            if result.returncode != 0:
                log_print(f"❌ Error creating chunk {i+1}")
                log_print(f"Error details: {result.stderr}")
                logging.error(f"Error creating chunk {i+1}: {result.stderr}")
            else:
                log_print(f"✓ Created chunk {i+1}: {os.path.basename(output_filename)}")
                log_print(f"  Processing time: {chunk_processing_time:.2f} seconds")
                
                # Add data to CSV
                append_to_csv(csv_file, num_chunks, i+1, chunk_duration, chunk_processing_time)
        
        total_operation_time = time.time() - operation_start_time
        log_print(f"\n✓ Successfully created {num_chunks} chunks!")
        log_print(f"Total processing time: {total_operation_time:.2f} seconds")
        log_print(f"Average time per chunk: {(total_chunk_time/num_chunks):.2f} seconds")
        log_print(f"{'='*50}\n")
        
        # Save terminal output
        save_terminal_output('\n'.join(terminal_output), run_dir, num_chunks)
        
    except subprocess.CalledProcessError as e:
        log_print(f"❌ Error processing video: {str(e)}")
        logging.error(f"Error processing video: {str(e)}")
    except Exception as e:
        log_print(f"❌ Unexpected error: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("\nVideo Chunking Tool")
    print("==================\n")
    
    # Create output directory for this run
    run_dir = setup_output_directory()
    csv_file = create_csv_file(run_dir)
    
    # Check for FFmpeg before proceeding
    check_ffmpeg()
    
    video_file = r"C:\Users\prady\Documents\Capstone\chunker-v0\UE18CS315_2020-09-11_CLASS25_SJ_11min.mp4"
    if not os.path.exists(video_file):
        logging.error(f"Video file not found: {video_file}")
        print(f"❌ Error: Video file not found: {video_file}")
        sys.exit(1)
    
    print(f"Starting processing for: {os.path.basename(video_file)}\n")
    
    for num_chunks in range(2, 11):
        chunk_video(video_file, num_chunks, run_dir, csv_file)
    
    print("All chunking operations completed successfully!")
