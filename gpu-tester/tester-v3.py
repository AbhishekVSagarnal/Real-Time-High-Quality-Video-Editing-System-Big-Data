import subprocess
import threading
import time
import psutil
import sys
import os

# Hardcoded input and output files
INPUT_FILE = "UE18CS315_2020-09-11_CLASS25_SJ_11min.mp4"
OUTPUT_FILE = "output_processed.mp4"

def get_metadata(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return None
        
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration,size,bit_rate', '-of', 'default=noprint_wrappers=1',
        file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error getting metadata: {str(e)}")
        return None

def process_video():
    cmd = [
        'ffmpeg', '-y',
        '-i', INPUT_FILE,
        '-c:v', 'h264_nvenc',  # NVIDIA GPU encoder
        '-preset', 'fast',      # Encoding speed preset
        '-b:v', '5M',          # Video bitrate
        '-vf', 'scale=1920:1080',  # Scale video
        '-c:a', 'copy',        # Copy audio without re-encoding
        OUTPUT_FILE
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return process
    except Exception as e:
        print(f"Error starting FFmpeg: {str(e)}")
        return None

def monitor_resources(stop_event):
    print("\n--- Resource Utilization ---")
    print("Time\tCPU%\tRAM%")
    
    while not stop_event.is_set():
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            ram_percent = psutil.virtual_memory().percent
            print(f"{time.strftime('%H:%M:%S')}\t{cpu_percent}%\t{ram_percent}%")
        except Exception as e:
            print(f"Error monitoring resources: {str(e)}")
        time.sleep(1)

def main():
    print("\n--- Starting Video Processing ---")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found!")
        return

    print("\n--- Metadata Before Processing ---")
    metadata_before = get_metadata(INPUT_FILE)
    if metadata_before:
        print(metadata_before)

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event,))
    monitor_thread.start()

    start_time = time.time()
    process = process_video()
    
    if process:
        stdout, stderr = process.communicate()
        end_time = time.time()
        stop_event.set()
        monitor_thread.join()

        print("\n--- Processing Output ---")
        print(stderr)

        if os.path.exists(OUTPUT_FILE):
            print("\n--- Metadata After Processing ---")
            metadata_after = get_metadata(OUTPUT_FILE)
            if metadata_after:
                print(metadata_after)

        print("\n--- Processing Time ---")
        print(f"Total Time: {end_time - start_time:.2f} seconds")
    else:
        stop_event.set()
        monitor_thread.join()
        print("Failed to start video processing")

if __name__ == "__main__":
    main()
