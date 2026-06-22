#!/usr/bin/env python3

import os
import sys
import subprocess
import tempfile
import re
import argparse
import shutil
import concurrent.futures
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import logging

def setup_logging():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,  # Or INFO to reduce verbosity
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr)  # Redirect logs to stderr
            ]
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description='Split a video into chunks, upload to HDFS, and process frames.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input video file (e.g., /path/to/video.mp4)')
    parser.add_argument('-n', '--num_chunks', type=int, required=True, help='Number of chunks to split the video into (positive integer)')
    parser.add_argument('-v', '--video_name', required=True, help='Name identifier for the video (used in HDFS paths)')
    return parser.parse_args()

def split_video(input_video, num_chunks, output_dir):
    """Split the video into specified number of chunks using ffmpeg."""
    LOGGER.debug(f"Starting video splitting for '{input_video}' with {num_chunks} chunks...")
    
    # Get total duration in seconds
    try:
        LOGGER.debug("Running ffprobe to get video duration...")
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', input_video],
            stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        total_duration = float(result.stdout.strip())
        LOGGER.debug(f"Total video duration obtained: {total_duration} seconds")
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"ffprobe error: {e.stderr}")
        sys.exit(1)
    except ValueError:
        LOGGER.error("Unable to parse video duration.")
        sys.exit(1)
    
    # Calculate segment duration
    seg_duration = total_duration / num_chunks
    LOGGER.debug(f"Each segment will be {seg_duration} seconds long")
    
    # Define output pattern
    output_pattern = os.path.join(output_dir, 'chunk_%03d.mp4')
    
    # Execute ffmpeg command to split the video
    try:
        LOGGER.debug("Executing ffmpeg to split the video...")
        result = subprocess.run(
            ['/usr/bin/ffmpeg', '-i', input_video,
             '-c', 'copy',
             '-map', '0',
             '-segment_time', f"{seg_duration}",
             '-f', 'segment',
             '-reset_timestamps', '1',
             output_pattern],
            stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        LOGGER.debug("Video successfully split into chunks.")
        LOGGER.debug(f"ffmpeg stdout: {result.stdout}")
        LOGGER.debug(f"ffmpeg stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"ffmpeg error: {e.stderr}")
        LOGGER.debug(f"ffmpeg stdout on error: {e.stdout}")
        sys.exit(1)
    
    # Verify number of chunks created
    chunks = sorted([f for f in os.listdir(output_dir) if f.startswith('chunk_') and f.endswith('.mp4')])
    actual_num_chunks = len(chunks)
    if actual_num_chunks != num_chunks:
        LOGGER.warning(f"Requested {num_chunks} chunks, but {actual_num_chunks} were created.")
    else:
        LOGGER.debug(f"Successfully created {actual_num_chunks} chunks.")
    
    return [os.path.join(output_dir, chunk) for chunk in chunks]


def create_hdfs_directory(base_dir, sub_dirs):
    """Create HDFS directories and set ACLs for user-2 and user-3."""
    for sub_dir in sub_dirs:
        full_path = os.path.join(base_dir, sub_dir)
        try:
            # Create the directory
            result = subprocess.run(['/opt/hadoop/bin/hadoop', 'fs', '-mkdir', '-p', full_path],
                                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            LOGGER.debug(f"Created HDFS directory: {full_path}")
            
            # Set ACLs for user-2 and user-3
            for user in ['user-2', 'user-3']:
                acl_result = subprocess.run(['/opt/hadoop/bin/hadoop', 'fs', '-setfacl', '-R', '-m', f"user:{user}:rwx", full_path],
                                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                LOGGER.debug(f"Set ACLs for {user} on {full_path}")
                if acl_result.stderr:
                    LOGGER.debug(f"HDFS ACL Error Output for {user}: {acl_result.stderr}")
            if result.stderr:
                LOGGER.debug(f"HDFS mkdir Error Output: {result.stderr}")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to create HDFS directory '{full_path}': {e.stderr}")
            sys.exit(1)


def upload_chunks_to_hdfs(chunk_files, hdfs_chunks_dir):
    """Upload video chunks to HDFS using multi-threading."""
    LOGGER.debug(f"Uploading {len(chunk_files)} chunks to HDFS directory '{hdfs_chunks_dir}'...")
    
    for chunk in chunk_files:
        LOGGER.debug(f"Attempting to upload: {chunk}")
    
    def upload_chunk(chunk_path):
        # Clean the chunk_path by removing any backticks or quotes
        chunk_path_clean = chunk_path.strip('`\'"')
        chunk_filename = os.path.basename(chunk_path_clean)
        LOGGER.debug(f"Uploading chunk '{chunk_filename}' from path: {chunk_path_clean}")
        try:
            result = subprocess.run(['/opt/hadoop/bin/hadoop', 'fs', '-put', '-f', chunk_path_clean, hdfs_chunks_dir],
                                    check=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
            LOGGER.debug(f"Uploaded '{chunk_filename}' to HDFS.")
            LOGGER.debug(f"HDFS put Output for '{chunk_filename}': {result.stdout}")
            if result.stderr:
                LOGGER.debug(f"HDFS put Error Output for '{chunk_filename}': {result.stderr}")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to upload '{chunk_filename}' to HDFS: {e.stderr}")
        except Exception as e:
            LOGGER.error(f"Unexpected error during upload of '{chunk_filename}': {e}")
    
    # Enable multi-threaded uploads
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(upload_chunk, chunk_files)
    
    LOGGER.debug("All chunks uploaded successfully.")

def list_hdfs_files(hdfs_dir):
    """List .mp4 files in the specified HDFS directory."""
    try:
        result = subprocess.run(
            ['/opt/hadoop/bin/hadoop', 'fs', '-ls', hdfs_dir],
            stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        files = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 8:
                filename = parts[-1]
                if filename.endswith('.mp4'):
                    files.append(filename)
        LOGGER.debug(f"Listed files in HDFS directory '{hdfs_dir}': {files}")
        return files
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Error listing HDFS directory {hdfs_dir}: {e.stderr}")
        return []

def get_frame_count_and_fps(hdfs_path):
    """Retrieve frame count and FPS of a video file using ffprobe."""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            local_video_filename = os.path.basename(hdfs_path)
            local_video_path = os.path.join(tmpdir, local_video_filename)

            # Copy video chunk from HDFS to local temporary directory
            subprocess.run(
                ['/opt/hadoop/bin/hadoop', 'fs', '-get', hdfs_path, local_video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            LOGGER.debug(f"Copied {hdfs_path} to {local_video_path}")

            # Use ffprobe to get duration and frame rate
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate,duration',
                '-of', 'default=nokey=1:noprint_wrappers=1',
                local_video_path
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            output = result.stdout.strip().split('\n')

            if len(output) >= 2:
                # Parse frame rate
                avg_frame_rate_str = output[0]
                match = re.match(r'(\d+)/(\d+)', avg_frame_rate_str)
                if match:
                    num, den = map(int, match.groups())
                    fps = num / den if den != 0 else 0
                else:
                    try:
                        fps = float(avg_frame_rate_str)
                    except ValueError:
                        fps = 0

                # Parse duration
                try:
                    duration = float(output[1])
                except ValueError:
                    duration = 0

                # Calculate frame count
                frame_count = int(round(duration * fps))
            else:
                frame_count = 0
                fps = 0

            LOGGER.debug(f"Video {hdfs_path}: Frame count = {frame_count}, FPS = {fps}")
            return frame_count, fps

    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Error executing ffprobe: {e.stderr}")
        return 0, 0
    except Exception as e:
        LOGGER.error(f"Unexpected error in get_frame_count_and_fps: {e}")
        return 0, 0

def process_video_file(index_and_info, video_name, hdfs_frame_dir):
    """Process a single video chunk: extract frames and upload to HDFS."""
    setup_logging()
    LOGGER = logging.getLogger(__name__)
    try:
        index, (hdfs_path, frame_offset) = index_and_info
        LOGGER.debug(f"Processing video chunk {index}: {hdfs_path} with frame offset {frame_offset}")
        
    except Exception as e:
        LOGGER.error(f"Error unpacking index_and_info: {e}")
        return

    def extract_and_upload_frames(tmpdir, local_video_path, frame_offset, archive_filename):
        setup_logging()
        
        LOGGER = logging.getLogger(__name__)
        try:
            # Use FFmpeg to extract frames
            frame_output_pattern = os.path.join(tmpdir, 'f%06d.jpg')
            LOGGER.debug(f"Extracting frames from {local_video_path} to {frame_output_pattern}")
            result = subprocess.run(
                ['/usr/local/bin/ffmpeg', '-i', local_video_path, '-qscale:v', '2', frame_output_pattern],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            LOGGER.debug(f"FFmpeg stdout: {result.stdout}")
            LOGGER.debug(f"FFmpeg stderr: {result.stderr}")

            if result.returncode != 0:
                LOGGER.error(f"FFmpeg failed with return code {result.returncode}")
                return

            LOGGER.debug(f"Successfully extracted frames from {local_video_path}")
        except Exception as e:
            LOGGER.error(f"Exception during FFmpeg execution: {e}")
            return

        # List extracted frames
        extracted_frames = sorted(
            [f for f in os.listdir(tmpdir) if f.startswith('f') and f.endswith('.jpg')]
        )
        LOGGER.debug(f"Extracted {len(extracted_frames)} frames from {hdfs_path}")

        if not extracted_frames:
            LOGGER.error(f"No frames extracted for {hdfs_path}")
            return

        #Rename frames with adjusted numbering
        renamed_frames = []
        for i, frame_filename in enumerate(extracted_frames, start=1):
            frame_counter = frame_offset + i
            new_frame_filename = f"frame{frame_counter:06d}.jpg"
            old_frame_path = os.path.join(tmpdir, frame_filename)
            new_frame_path = os.path.join(tmpdir, new_frame_filename)
            try:
                os.rename(old_frame_path, new_frame_path)
                renamed_frames.append(new_frame_filename)
            except Exception as e:
                LOGGER.error(f"Error renaming {old_frame_path} to {new_frame_path}: {e}")
                continue

        if not renamed_frames:
            LOGGER.error(f"No frames found after renaming for {hdfs_path}")
            return

        LOGGER.debug(f"Renamed frames: {renamed_frames}")

        # Archive frames using tar
        archive_path = os.path.join(tmpdir, archive_filename)
        try:
            LOGGER.debug(f"Creating archive {archive_path}")
            tar_command = ['tar', '-czf', archive_path, '-C', tmpdir] + renamed_frames
            result = subprocess.run(
                tar_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            LOGGER.debug(f"Tar stdout: {result.stdout}")
            LOGGER.debug(f"Tar stderr: {result.stderr}")

            if result.returncode != 0:
                LOGGER.error(f"Tar failed with return code {result.returncode}")
                return

            LOGGER.debug(f"Created archive {archive_path}")
        except Exception as e:
            LOGGER.error(f"Exception during tar execution: {e}")
            return

        # Upload archive to HDFS
        try:
            LOGGER.debug(f"Uploading archive {archive_path} to HDFS directory {hdfs_frame_dir}")
            result = subprocess.run(
                ['/opt/hadoop/bin/hadoop', 'fs', '-put', '-f', archive_path, hdfs_frame_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            LOGGER.debug(f"Uploaded archive to {hdfs_frame_dir}")
            LOGGER.debug(f"HDFS put archive Output: {result.stdout}")
            if result.stderr:
                LOGGER.debug(f"HDFS put archive Error Output: {result.stderr}")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to upload archive to HDFS: {e.stderr}")
            return
        except Exception as e:
            LOGGER.error(f"Exception during HDFS put: {e}")
            return

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            local_video_filename = os.path.basename(hdfs_path)
            local_video_path = os.path.join(tmpdir, local_video_filename)

            LOGGER.debug(f"Downloading {hdfs_path} to {local_video_path}")
            try:
                result = subprocess.run(
                    ['/opt/hadoop/bin/hadoop', 'fs', '-get', hdfs_path, local_video_path],
                    stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                    text=True
                )
                LOGGER.debug(f"HDFS get stdout: {result.stdout}")
                LOGGER.debug(f"HDFS get stderr: {result.stderr}")

                if result.returncode != 0:
                    LOGGER.error(f"HDFS get failed with return code {result.returncode}")
                    return

                LOGGER.debug(f"Retrieved {hdfs_path} to {local_video_path}")
            except Exception as e:
                LOGGER.error(f"Exception during HDFS get: {e}")
                return  # Exit the function

            # Get frame count and fps
            frame_count, fps = get_frame_count_and_fps(hdfs_path)
            LOGGER.debug(f"Video {hdfs_path}: Frame count = {frame_count}, FPS = {fps}")

            # Generate a unique archive filename based on chunk index or name
            chunk_basename = os.path.splitext(os.path.basename(hdfs_path))[0]  # e.g., 'chunk_000'
            archive_filename = f"{chunk_basename}_frames.tar.gz"

            # Extract and upload frames
            extract_and_upload_frames(tmpdir, local_video_path, frame_offset, archive_filename)

            LOGGER.debug(f"Finished processing video: {hdfs_path}")

    except Exception as e:
        LOGGER.error(f"Unexpected error in process_video_file: {e}")
        raise e  # Re-raise exception to indicate failure

def main():
    setup_logging()
    LOGGER = logging.getLogger(__name__)

    args = parse_arguments()
    LOGGER.debug("Starting the main process with input arguments.")
    input_video = args.input
    num_chunks = args.num_chunks
    video_name = args.video_name
    
    # Validate number of chunks
    if num_chunks < 1:
        LOGGER.error("Number of chunks must be at least 1.")
        sys.exit(1)
    LOGGER.debug("Setting up Spark configuration...")
    # Initialize Spark Configuration with optimized settings
    conf = SparkConf() \
    .setAppName("VideoChunkProcessor") \
    .setMaster("spark://user:7077") \
    .set("spark.executor.instances", "10") \
    .set("spark.executor.memory", "1g") \
    .set("spark.executor.memoryOverhead", "512m") \
    .set("spark.executor.cores", "4") \

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext 
    
    try:
        # Create temporary directory for splitting
        with tempfile.TemporaryDirectory() as temp_dir:
            LOGGER.debug(f"Created temporary directory at '{temp_dir}'")
            
            # Split the video into chunks
            chunk_files = split_video(input_video, num_chunks, temp_dir)
            
            # Log chunk files and verify their existence
            LOGGER.debug(f"Chunk files created: {chunk_files}")
            for chunk in chunk_files:
                if not os.path.isfile(chunk):
                    LOGGER.error(f"Chunk file does not exist: {chunk}")
                else:
                    LOGGER.debug(f"Chunk file exists: {chunk}")
            
            # Define HDFS base directory
            hdfs_base_dir = f'/user/workspace/{video_name}'
            
            # Define HDFS chunks and frames directories
            hdfs_chunks_dir = os.path.join(hdfs_base_dir, 'chunks')
            hdfs_frames_dir = os.path.join(hdfs_base_dir, 'frames')
            
            # Create HDFS directories
            create_hdfs_directory(hdfs_base_dir, ['chunks', 'frames'])
            
            # Upload chunks to HDFS 
            upload_chunks_to_hdfs(chunk_files, hdfs_chunks_dir)
    
    except Exception as e:
        LOGGER.error(f"Error during video splitting and uploading: {e}")
        spark.stop()
        sys.exit(1)
    
    LOGGER.debug("Video splitting and uploading completed successfully.")
    
    try:
        # Define HDFS chunks directory
        video_chunks_path = hdfs_chunks_dir
    
        # List video chunk files in HDFS
        video_files = list_hdfs_files(video_chunks_path)
        if not video_files:
            LOGGER.error("No video files found in HDFS directory.")
            spark.stop()
            sys.exit(1)
    
        # Parallelize frame count computations
        video_files_rdd = sc.parallelize(video_files, numSlices=num_chunks)
    
        def get_frame_info(hdfs_filename):
            hdfs_path = os.path.join(video_chunks_path, hdfs_filename)
            frame_count, fps = get_frame_count_and_fps(hdfs_path)
            return (hdfs_path, frame_count, fps)
    
        frame_info_rdd = video_files_rdd.map(get_frame_info)
        frame_info_list = frame_info_rdd.collect()
    
        # Process frame counts and calculate frame offsets
        frame_counts = []
        total_frames = 0
        for hdfs_path, frame_count, fps in frame_info_list:
            frame_counts.append((hdfs_path, frame_count))
            total_frames += frame_count
            LOGGER.debug(f"Video file: {hdfs_path}, Frame count: {frame_count}, FPS: {fps}")
    
        # Calculate frame offsets
        cumulative_frames = [0]
        for _, count in frame_counts[:-1]:
            cumulative_frames.append(cumulative_frames[-1] + count)
    
        # Pair each video file with its frame offset
        video_info = []
        for idx, ((hdfs_path, frame_count), frame_offset) in enumerate(zip(frame_counts, cumulative_frames)):
            video_info.append((idx, (hdfs_path, frame_offset)))
    
        # Create an RDD from the list of video info
        video_info_rdd = sc.parallelize(video_info, numSlices=num_chunks)
    
        # Broadcast video_name and hdfs_frames_dir to executors
        video_name_broadcast = sc.broadcast(video_name)
        hdfs_frames_dir_broadcast = sc.broadcast(hdfs_frames_dir)
    
        # Modify process_video_file to accept video_name and hdfs_frames_dir via closure
        def process_video_file_wrapper(index_and_info):
            process_video_file(index_and_info, video_name_broadcast.value, hdfs_frames_dir_broadcast.value)
    
        # Process each video chunk in parallel to extract frames
        video_info_rdd.foreach(process_video_file_wrapper)
    
        LOGGER.debug("Frame extraction and uploading completed successfully.")
    
    except Exception as e:
        LOGGER.error(f"Error during frame extraction and uploading: {e}")
    finally:
        # Stop Spark Session
        spark.stop()
        LOGGER.debug("Spark session stopped.")

if __name__ == "__main__":
    main()
