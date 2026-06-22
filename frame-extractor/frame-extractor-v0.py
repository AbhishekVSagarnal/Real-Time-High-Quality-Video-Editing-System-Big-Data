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
from tqdm import tqdm

class VideoFrameExtractor:
    def __init__(self, input_video, output_dir):
        """Initialize the frame extractor with paths and create directories."""
        self.input_video = input_video
        self.input_video_name = Path(input_video).stem  # Get filename without extension
        self.output_dir = Path(output_dir) / 'frame-extractor-v0-output'
        self.frames_dir = self.output_dir / 'frame-extractor-v0-frames'
        self.metrics_file = self.output_dir / 'frame-extractor-v0_metrics.json'
        self.stats_file = self.output_dir / 'frame-extractor-v0_stats.txt'
        
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
        
        self.update_stats("Initialization Complete")

    def update_stats(self, message):
        """Update the stats file with new information."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.stats_file, 'a') as f:
            f.write(f"\n[{timestamp}] {message}")
            f.write(f"\nCPU Usage: {psutil.cpu_percent()}%")
            f.write(f"\nMemory Usage: {psutil.Process().memory_info().rss/1024/1024:.1f}MB")
            f.write("\n" + "="*50 + "\n")

    def get_video_info(self):
        """Analyze video and extract information."""
        print("\nPHASE 1: VIDEO ANALYSIS")
        print("="*50)
        
        cap = cv2.VideoCapture(self.input_video)
        video_info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            'file_size_mb': os.path.getsize(self.input_video) / (1024 * 1024)
        }
        cap.release()
        
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
        """Extract and save audio from video."""
        print("\nPHASE 2: AUDIO EXTRACTION")
        print("="*50)
        
        start_time = time.time()
        audio_path = self.output_dir / f'{self.input_video_name}_audio-extract.aac'
        
        cmd = [
            'ffmpeg', '-i', self.input_video,
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
        """Extract frames from video."""
        print("\nPHASE 3: FRAME EXTRACTION")
        print("="*50)
        
        start_time = time.time()
        
        cmd = [
            'ffmpeg', '-i', self.input_video,
            '-vf', f"fps={self.metrics['video_metrics']['fps']}",
            str(self.frames_dir / 'frame_%06d.png')
        ]
        
        with tqdm(total=self.metrics['video_metrics']['frame_count'], 
                 desc="Extracting frames") as pbar:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            pbar.update(self.metrics['video_metrics']['frame_count'])
        
        extracted_frames = list(self.frames_dir.glob('*.png'))
        extraction_time = time.time() - start_time
        
        # Calculate frame sizes
        frame_sizes = [f.stat().st_size for f in extracted_frames]
        
        frame_metrics = {
            'extraction_time': extraction_time,
            'frames_extracted': len(extracted_frames),
            'avg_time_per_frame': extraction_time / len(extracted_frames),
            'total_size_mb': sum(frame_sizes) / (1024 * 1024),
            'avg_frame_size_kb': (sum(frame_sizes) / len(frame_sizes)) / 1024 if frame_sizes else 0
        }
        
        self.metrics['frame_extraction'] = frame_metrics
        
        print(f"\nFrame Extraction Results:")
        print(f"├── Frames Extracted: {frame_metrics['frames_extracted']}")
        print(f"├── Total Size: {frame_metrics['total_size_mb']:.2f}MB")
        print(f"├── Average Frame Size: {frame_metrics['avg_frame_size_kb']:.2f}KB")
        print(f"├── Total Time: {frame_metrics['extraction_time']:.2f}s")
        print(f"└── Average Time per Frame: {frame_metrics['avg_time_per_frame']*1000:.2f}ms")
        
        self.update_stats(f"Frame Extraction Complete\n{json.dumps(frame_metrics, indent=2)}")
        return frame_metrics

    def process(self):
        """Main processing pipeline."""
        print("\n" + "="*80)
        print("VIDEO FRAME EXTRACTION - DETAILED MONITORING")
        print("="*80)
        
        overall_start = time.time()
        
        try:
            # Phase 1: Video Analysis
            self.get_video_info()
            
            # Phase 2: Audio Extraction
            self.extract_audio()
            
            # Phase 3: Frame Extraction
            self.extract_frames()
            
            # Final Summary
            total_time = time.time() - overall_start
            self.metrics['overall'] = {
                'total_time': total_time,
                'peak_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'final_cpu_percent': psutil.cpu_percent()
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
            print(f"\n❌ Error: {str(e)}")
            self.update_stats(f"Error occurred: {str(e)}")
            raise

def main():
    input_video = r"C:\Users\prady\Documents\Capstone\frame-extractor\UE18CS315_2020-09-11_CLASS25_SJ.mp4"
    output_dir = str(Path(input_video).parent)  # Save in same directory as input video
    
    print("\nInitializing Frame Extractor...")
    print(f"Input Video: {input_video}")
    print(f"Output Directory: {output_dir}")
    
    extractor = VideoFrameExtractor(input_video, output_dir)
    extractor.process()

if __name__ == "__main__":
    main() 