#!/usr/bin/env python3
import os
import time
import sys
import subprocess
import tempfile
import re
import argparse
import shutil
import concurrent.futures
import json
from fractions import Fraction
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
LOGGER = logging.getLogger(__name__)

def split_video(input_video, num_chunks, output_dir):
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                input_video
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        total_duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, Exception) as e:
        raise Exception(f"An unexpected error occurred: {e}")
    
    seg_duration = total_duration / num_chunks
    output_pattern = os.path.join(output_dir, 'chunk_%03d.mp4')
    
    try:
        result = subprocess.run(
            [
                'ffmpeg',
                '-i', input_video,
                '-c', 'copy',
                '-map', '0',
                '-segment_time', f"{seg_duration}",
                '-f', 'segment',
                '-reset_timestamps', '1',
                output_pattern
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except (subprocess.CalledProcessError, Exception) as e:
        raise Exception(f"An unexpected error occurred: {e}")
    
    chunk_files = sorted(
        [
            f for f in os.listdir(output_dir)
            if f.startswith('chunk_') and f.endswith('.mp4')
        ]
    )

    chunks_info = []
    for chunk_file in chunk_files:
        chunk_path = os.path.join(output_dir, chunk_file)
        try:
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-count_frames',
                    '-show_entries', 'stream=nb_read_frames,avg_frame_rate',
                    '-of', 'json',
                    chunk_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            ffprobe_output = json.loads(result.stdout)
            stream_info = ffprobe_output['streams'][0]
            nb_frames = int(stream_info.get('nb_read_frames', 0))
            avg_frame_rate = stream_info.get('avg_frame_rate', '0/1')
            fps = float(Fraction(avg_frame_rate))

            LOGGER.info(f"Chunk: {chunk_path}, Frames: {nb_frames}, FPS: {fps}")
            
            chunks_info.append({
                'chunk_path': chunk_path,
                'frame_count': nb_frames,
                'fps': fps
            })
        except (subprocess.CalledProcessError, Exception) as e:
            raise Exception(f"An unexpected error occurred: {e}")
    
    return chunks_info

def create_hdfs_directory(base_dir, sub_dirs, workers):
    for sub_dir in sub_dirs:
        full_path = os.path.join(base_dir, sub_dir)
        try:
            result = subprocess.run(['/opt/hadoop/bin/hadoop', 'fs', '-mkdir', '-p', full_path],
                                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for user in workers:
                acl_result = subprocess.run(['/opt/hadoop/bin/hadoop', 'fs', '-setfacl', '-R', '-m', f"user:{user}:rwx", full_path],
                                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        except (subprocess.CalledProcessError, Exception) as e:
            raise Exception(f"Failed to create HDFS directory '{full_path}': {e}")

def upload_chunks_to_hdfs(chunk_files, hdfs_chunks_dir, num_chunks):
    LOGGER.info(f"Uploading {len(chunk_files)} chunks to HDFS")
    
    def upload_chunk(chunk_path):
        chunk_path_clean = chunk_path.strip('`\'"')
        chunk_filename = os.path.basename(chunk_path_clean)
        try:
            result = subprocess.run(['/opt/hadoop/bin/hadoop', 'fs', '-put', '-f', chunk_path_clean, hdfs_chunks_dir],
                                    check=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
        except (subprocess.CalledProcessError, Exception) as e:
            raise Exception(f"Failed to upload '{chunk_filename}' to HDFS: {e}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
        executor.map(upload_chunk, chunk_files)

def process_video_file(index_and_info, input_video, hdfs_frame_dir):
    idx, (hdfs_path, frame_offset) = index_and_info

    def extract_and_upload_frames(tmpdir, local_video_path, frame_offset, archive_filename):
        try:
            frame_output_pattern = os.path.join(tmpdir, 'f%06d.jpg')
            result = subprocess.run(
                ['/usr/local/bin/ffmpeg', '-hwaccel', 'cuda', '-c:v', 'h264_cuvid', '-i', local_video_path, '-qscale:v', '2', frame_output_pattern],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg failed with return code {result.returncode}")

            LOGGER.info(f"Successfully extracted frames from {local_video_path}")
        except Exception as e:
            raise Exception(f"Exception during FFmpeg execution: {e}")

        extracted_frames = sorted(
            [f for f in os.listdir(tmpdir) if f.startswith('f') and f.endswith('.jpg')]
        )
        if not extracted_frames:
            raise Exception(f"No frames extracted for {local_video_path}")

        # renamed_frames = []

        # for i, frame_filename in enumerate(extracted_frames, start=1):
        #     frame_counter = frame_offset + i
        #     new_frame_filename = f"frame{frame_counter:06d}.jpg"
        #     old_frame_path = os.path.join(tmpdir, frame_filename)
        #     new_frame_path = os.path.join(tmpdir, new_frame_filename)
        #     try:
        #         os.rename(old_frame_path, new_frame_path)
        #         renamed_frames.append(new_frame_filename)
        #     except Exception as e:
        #         raise Exception(f"Error renaming {old_frame_path} to {new_frame_path}: {e}")

        # if not renamed_frames:
        #     raise Exception(f"No frames found after renaming for {local_video_path}")

        # LOGGER.info(f"Renamed frames: {renamed_frames}")

        # archive_path = os.path.join(tmpdir, archive_filename)
        # try:
        #     LOGGER.info(f"Creating archive {archive_path}")
        #     tar_command = ['tar', '-czf', archive_path, '-C', tmpdir] + renamed_frames
        #     result = subprocess.run(
        #         tar_command,
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.PIPE,
        #         text=True,
        #         check=True
        #     )

        #     if result.returncode != 0:
        #         raise Exception(f"Tar failed with return code {result.returncode}")

        #     LOGGER.info(f"Created archive {archive_path}")
        # except Exception as e:
        #     raise Exception(f"Exception during tar execution: {e}")

        # try:
        #     LOGGER.info(f"Uploading archive {archive_path} to HDFS directory {hdfs_frame_dir}")
        #     result = subprocess.run(
        #         ['/opt/hadoop/bin/hadoop', 'fs', '-put', '-f', archive_path, hdfs_frame_dir],
        #         check=True,
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.PIPE,
        #         text=True
        #     )
        # except (subprocess.CalledProcessError, Exception) as e:
        #     raise Exception(f"Exception during HDFS put: {e}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_video_filename = os.path.basename(hdfs_path)
            local_video_path = os.path.join(tmpdir, local_video_filename)

            LOGGER.info(f"Downloading {hdfs_path} to {local_video_path}")
            try:
                result = subprocess.run(
                    ['/opt/hadoop/bin/hadoop', 'fs', '-get', hdfs_path, local_video_path],
                    stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )

                if result.returncode != 0:
                    raise Exception(f"HDFS get failed with return code {result.returncode}")

            except Exception as e:
                raise Exception(f"Exception during HDFS get: {e}")

            chunk_basename = os.path.splitext(os.path.basename(hdfs_path))[0]
            archive_filename = f"{chunk_basename}_frames.tar.gz"

            extract_and_upload_frames(tmpdir, local_video_path, frame_offset, archive_filename)

    except Exception as e:
        raise Exception(f"Unexpected error in process_video_file: {e}")

def create_chunks_and_upload(input_video, num_chunks, hdfs_base_dir, workers, hdfs_chunks_dir):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks_info = split_video(input_video, num_chunks, temp_dir)
            chunk_files = [chunk['chunk_path'] for chunk in chunks_info]
            
            create_hdfs_directory(hdfs_base_dir, ['chunks', 'frames'], workers)
            upload_chunks_to_hdfs(chunk_files, hdfs_chunks_dir, num_chunks)

            return chunks_info
    except Exception as e:
        raise Exception(f"Error during video splitting and uploading: {e}")

def main():
    start_time = time.time()
    input_video = "3-hr.mp4"
    num_chunks = 10
    workers = ['user-2', 'user-3']
    
    conf = SparkConf() \
    .setAppName(f"Video Processing | {input_video} | Chunks: {num_chunks} | GPU") \
    .setMaster("spark://user:7077") \
    .set("spark.executor.instances", "10") \
    .set("spark.executor.memory", "1g") \
    .set("spark.executor.memoryOverhead", "512m") \
    .set("spark.executor.cores", "4") \

    hdfs_base_dir = f'/user/workspace/{input_video}'
    hdfs_chunks_dir = os.path.join(hdfs_base_dir, 'chunks')
    hdfs_frames_dir = os.path.join(hdfs_base_dir, 'frames')

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext 

    try:
        LOGGER.info("Starting video splitting and uploading...")

        chunks_info = create_chunks_and_upload(input_video, num_chunks, hdfs_base_dir, workers, hdfs_chunks_dir)

        LOGGER.info(f"{'*'*100}")
        chunking_time = time.time() - start_time
        LOGGER.info(f"Time Taken for Chunking and Uploading: {chunking_time} seconds")
        LOGGER.info(f"{'*'*100}")

        LOGGER.info("Starting frame extraction and uploading...")

        frame_counts = []
        total_frames = 0

        for info in chunks_info:
            chunk_filename = os.path.basename(info['chunk_path'])
            hdfs_path = os.path.join(hdfs_chunks_dir, chunk_filename)
            frame_count = info['frame_count']
            fps = info['fps']
            frame_counts.append((hdfs_path, frame_count))
            total_frames += frame_count

        cumulative_frames = [0]
        for _, count in frame_counts[:-1]:
            cumulative_frames.append(cumulative_frames[-1] + count)

        video_info = []
        for idx, ((hdfs_path, frame_count), frame_offset) in enumerate(zip(frame_counts, cumulative_frames)):
            video_info.append((idx, (hdfs_path, frame_offset)))

        video_info_rdd = sc.parallelize(video_info, numSlices=num_chunks)

        video_name_broadcast = sc.broadcast(input_video)
        hdfs_frames_dir_broadcast = sc.broadcast(hdfs_frames_dir)

        def process_video_file_wrapper(index_and_info):
            process_video_file(index_and_info, video_name_broadcast.value, hdfs_frames_dir_broadcast.value)

        video_info_rdd.foreach(process_video_file_wrapper)

        extraction_time = time.time() - start_time - chunking_time
        LOGGER.info(f"Time Taken for Frame Extraction and Upload: {extraction_time} seconds")
        LOGGER.info(f"{'*'*100}")   

    except Exception as e:
        raise Exception(f"Error during frame extraction and uploading: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()