import cv2
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime
import logging
import os
import json
import time
import psutil
import threading
from tqdm import tqdm

class GPUVideoFrameExtractor:
    def __init__(self, input_video, output_dir):
        """Initialize the GPU-accelerated frame extractor with paths and create directories."""
        self.input_video = input_video
        self.input_video_name = Path(input_video).stem
        
        # Create timestamp for unique folder identification
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create version-specific output directories with timestamp
        self.output_dir = Path(output_dir) / f'frame-extractor-v1-output_{timestamp}'
        self.frames_dir = self.output_dir / f'frame-extractor-v1-frames_{timestamp}'
        self.metrics_file = self.output_dir / f'frame-extractor-v1_metrics_{timestamp}.json'
        self.stats_file = self.output_dir / f'frame-extractor-v1_stats_{timestamp}.txt'
        
        # Create directories
        for dir_path in [self.output_dir, self.frames_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'video_metrics': {},
            'audio_extraction': {},
            'frame_extraction': {},
            'system_metrics': {
                'initial_memory': psutil.Process().memory_info().rss / 1024 / 1024,
                'initial_cpu': psutil.cpu_percent()
            }
        }
        
        # Resource monitoring
        self.stop_monitoring = threading.Event()
        self.resource_metrics = []
        
        self.update_stats("Initialization Complete")

    def monitor_resources(self):
        """Monitor system resources during processing."""
        while not self.stop_monitoring.is_set():
            metrics = {
                'timestamp': time.strftime('%H:%M:%S'),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'ram_percent': psutil.virtual_memory().percent
            }
            self.resource_metrics.append(metrics)
            time.sleep(1)

    def update_stats(self, message):
        """Update the stats file with new information."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.stats_file, 'a') as f:
            f.write(f"\n[{timestamp}] {message}")
            f.write(f"\nCPU Usage: {psutil.cpu_percent()}%")
            f.write(f"\nMemory Usage: {psutil.Process().memory_info().rss/1024/1024:.1f}MB")
            f.write("\n" + "="*50 + "\n")

    def get_video_info(self):
        """Analyze video and extract information using ffprobe."""
        print("\nPHASE 1: VIDEO ANALYSIS")
        print("="*50)
        
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
            '-show_entries', 'format=duration,size',
            '-of', 'json',
            self.input_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        video_stream = next(s for s in data['streams'] if 'width' in s)
        fps_parts = video_stream['r_frame_rate'].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1])
        
        video_info = {
            'fps': fps,
            'frame_count': int(video_stream.get('nb_frames', 0)),
            'width': video_stream['width'],
            'height': video_stream['height'],
            'duration': float(data['format']['duration']),
            'file_size_mb': float(data['format']['size']) / (1024 * 1024)
        }
        
        self.metrics['video_metrics'] = video_info
        
        print(f"\nVideo Analysis Results:")
        print(f"├── Resolution: {video_info['width']}x{video_info['height']}")
        print(f"├── FPS: {video_info['fps']:.2f}")
        print(f"├── Frame Count: {video_info['frame_count']}")
        print(f"├── Duration: {video_info['duration']:.2f}s")
        print(f"└── File Size: {video_info['file_size_mb']:.2f}MB")
        
        self.update_stats(f"Video Analysis Complete\n{json.dumps(video_info, indent=2)}")
        return video_info

    def extract_audio(self):
        """Extract and save audio using GPU-accelerated processing."""
        print("\nPHASE 2: AUDIO EXTRACTION")
        print("="*50)
        
        start_time = time.time()
        audio_path = self.output_dir / f'{self.input_video_name}_audio-extract.aac'
        
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',  # Enable CUDA hardware acceleration
            '-i', self.input_video,
            '-vn', '-acodec', 'copy',
            str(audio_path)
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        audio_metrics = {
            'extraction_time': time.time() - start_time,
            'success': process.returncode == 0,
            'audio_found': audio_path.exists()
        }
        
        if audio_path.exists():
            audio_metrics['audio_size_mb'] = audio_path.stat().st_size / (1024 * 1024)
            print(f"\nAudio Extraction Complete:")
            print(f"├── Audio Size: {audio_metrics['audio_size_mb']:.2f}MB")
            print(f"└── Extraction Time: {audio_metrics['extraction_time']:.2f}s")
        else:
            print("\n⚠️ No audio found in video or extraction failed")
        
        self.metrics['audio_extraction'] = audio_metrics
        self.update_stats(f"Audio Extraction Complete\n{json.dumps(audio_metrics, indent=2)}")

    def extract_frames(self):
        """Extract frames using GPU acceleration only."""
        print("\nPHASE 3: GPU-ACCELERATED FRAME EXTRACTION")
        print("="*50)
        
        start_time = time.time()
        
        # GPU-only command using NVIDIA hardware acceleration
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda',
            '-i', str(self.input_video),
            '-vf', f"fps={self.metrics['video_metrics']['fps']}",
            '-c:v', 'h264_nvenc',  # Use NVIDIA encoder
            '-preset', 'p7',        # Highest quality preset
            '-qp', '0',            # Lossless quality
            '-f', 'image2',        # Force image sequence output
            str(self.frames_dir / 'frame_%06d.png')
        ]
        
        print("\nExecuting GPU-only extraction...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        with tqdm(total=self.metrics['video_metrics']['frame_count'],
                 desc="Extracting frames (GPU)") as pbar:
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Update progress bar occasionally
                    pbar.update(min(100, self.metrics['video_metrics']['frame_count'] - pbar.n))
        
        stdout, stderr = process.communicate()
        
        # Error check
        if process.returncode != 0:
            print("\nFFmpeg GPU Error Output:")
            print(stderr)
            raise Exception(f"GPU frame extraction failed with return code {process.returncode}")
        
        # Wait for filesystem
        time.sleep(2)
        
        # Get extracted frames
        extracted_frames = list(self.frames_dir.glob('*.png'))
        
        if not extracted_frames:
            raise Exception("No frames were extracted! Check FFmpeg output for errors.")
        
        extraction_time = time.time() - start_time
        
        # Calculate frame sizes
        frame_sizes = [f.stat().st_size for f in extracted_frames]
        
        frame_metrics = {
            'extraction_time': extraction_time,
            'frames_extracted': len(extracted_frames),
            'avg_time_per_frame': extraction_time / len(extracted_frames) if extracted_frames else 0,
            'total_size_mb': sum(frame_sizes) / (1024 * 1024),
            'avg_frame_size_kb': (sum(frame_sizes) / len(frame_sizes)) / 1024 if frame_sizes else 0,
            'ffmpeg_returncode': process.returncode
        }
        
        self.metrics['frame_extraction'] = frame_metrics
        
        print(f"\nGPU Frame Extraction Results:")
        print(f"├── Frames Extracted: {frame_metrics['frames_extracted']}")
        print(f"├── Total Size: {frame_metrics['total_size_mb']:.2f}MB")
        print(f"├── Average Frame Size: {frame_metrics['avg_frame_size_kb']:.2f}KB")
        print(f"├── Total Time: {frame_metrics['extraction_time']:.2f}s")
        print(f"└── Average Time per Frame: {frame_metrics['avg_time_per_frame']*1000:.2f}ms")
        
        self.update_stats(f"GPU Frame Extraction Complete\n{json.dumps(frame_metrics, indent=2)}")
        return frame_metrics

    def process(self):
        """Main processing pipeline with resource monitoring."""
        print("\n" + "="*80)
        print("GPU-ACCELERATED VIDEO FRAME EXTRACTION - DETAILED MONITORING")
        print("="*80)
        
        overall_start = time.time()
        
        # Start resource monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_resources)
        monitor_thread.start()
        
        try:
            # Phase 1: Video Analysis
            self.get_video_info()
            
            # Phase 2: Audio Extraction
            self.extract_audio()
            
            # Phase 3: Frame Extraction
            self.extract_frames()
            
            # Stop resource monitoring
            self.stop_monitoring.set()
            monitor_thread.join()
            
            # Final Summary
            total_time = time.time() - overall_start
            self.metrics['overall'] = {
                'total_time': total_time,
                'peak_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'final_cpu_percent': psutil.cpu_percent(),
                'resource_monitoring': self.resource_metrics
            }
            
            print("\nFINAL PROCESSING SUMMARY")
            print("="*50)
            print(f"Overall Metrics:")
            print(f"├── Total Processing Time: {total_time:.2f}s")
            print(f"├── Peak Memory Usage: {self.metrics['overall']['peak_memory_mb']:.1f}MB")
            print(f"└── Final CPU Usage: {self.metrics['overall']['final_cpu_percent']}%")
            
            # Save final metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            self.update_stats("Processing Complete")
            return True
            
        except Exception as e:
            self.stop_monitoring.set()
            monitor_thread.join()
            print(f"\n❌ Error: {str(e)}")
            self.update_stats(f"Error occurred: {str(e)}")
            raise

def main():
    input_video = r"C:\Users\prady\Documents\Capstone\frame-extractor\UE18CS315_2020-09-11_CLASS25_SJ.mp4"
    output_dir = str(Path(input_video).parent)
    
    print("\nInitializing GPU-Accelerated Frame Extractor...")
    print(f"Input Video: {input_video}")
    print(f"Output Directory: {output_dir}")
    
    extractor = GPUVideoFrameExtractor(input_video, output_dir)
    extractor.process()

if __name__ == "__main__":
    main() 