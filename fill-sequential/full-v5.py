# A GPU-accelerated video processing pipeline that extracts frames using FFmpeg+CUDA, detects logo position once using OpenCV CPU template matching, replaces logos in parallel using FFmpeg+CUDA overlay filters, and recompiles the video using NVIDIA hardware encoding, with comprehensive progress monitoring and metrics tracking.

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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import traceback

class GPUVideoLogoReplacer:
    def __init__(self, input_video, old_logo, new_logo, output_dir="output"):
        """Initialize with GPU acceleration and multi-threading support."""
        self.input_video = input_video
        self.old_logo = old_logo
        self.new_logo = new_logo

        # Create timestamp for unique folder identification
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Update directory names to include version and timestamp
        self.output_dir = Path(output_dir) / f'full-v5-output_{timestamp}'
        self.frames_dir = self.output_dir / f'frames_full-v5_{timestamp}'
        self.processed_dir = self.output_dir / f'processed_frames_full-v5_{timestamp}'
        self.checkpoint_dir = self.output_dir / f'checkpoints_full-v5_{timestamp}'
        self.metrics_file = self.output_dir / f'full-v5_metrics_{timestamp}.json'
        self.stats_file = self.output_dir / f'full-v5_stats_{timestamp}.txt'

        # Create directories
        for dir_path in [self.output_dir, self.frames_dir, self.processed_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Get CPU count for optimal thread allocation
        self.cpu_count = multiprocessing.cpu_count()

        # Initialize thread-safe metrics storage
        self.metrics_lock = threading.Lock()
        self.metrics = {
            'video_metrics': {},
            'checkpoints': {
                'last_successful_phase': None,
                'frames_processed': 0
            },
            'overall_metrics': {
                'start_time': datetime.now().isoformat(),
                'cpu_usage': psutil.cpu_percent(),
                'initial_memory': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_count': self.cpu_count
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
            f.write(f"\nThreads Used: {self.cpu_count}")
            f.write("\n" + "="*50 + "\n")

    def get_video_info(self):
        """Analyze video and extract information using ffprobe."""
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
        return video_info



    def detect_initial_logo(self, first_frame_path):
        """Detect logo in first frame using OpenCV template matching."""
        print("\nPHASE 3.1: INITIAL LOGO DETECTION")
        print("="*50)
        
        start_time = time.time()
        frame = cv2.imread(str(first_frame_path))
        template = cv2.imread(str(self.old_logo))
        
        if frame is None:
            raise ValueError(f"Failed to load frame: {first_frame_path}")
        if template is None:
            raise ValueError(f"Failed to load template: {self.old_logo}")
            
        print(f"\nAnalyzing first frame for logo detection:")
        print(f"├── Frame shape: {frame.shape}")
        print(f"└── Template shape: {template.shape}")
        
        best_confidence = 0
        best_loc = None
        best_size = None
        best_scale = None
        
        # Try multiple scales for better detection
        scales = np.linspace(0.8, 2.0, 25)
        
        for scale in scales:
            width = int(template.shape[1] * scale)
            height = int(template.shape[0] * scale)
            resized_template = cv2.resize(template, (width, height))
            
            result = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_loc = max_loc
                best_size = (width, height)
                best_scale = scale
        
        detection_time = time.time() - start_time
        
        # Save detection results
        detection_info = {
            'success': best_confidence > 0.3,
            'confidence': float(best_confidence),
            'location': best_loc,
            'size': best_size,
            'scale': float(best_scale),
            'detection_time': detection_time
        }
        
        print(f"\nLogo Detection Results:")
        print(f"├── Confidence: {best_confidence:.4f}")
        print(f"├── Location: {best_loc}")
        print(f"├── Size: {best_size}")
        print(f"├── Scale: {best_scale:.2f}")
        print(f"└── Detection Time: {detection_time*1000:.2f}ms")
        
        # Save debug image
        if detection_info['success']:
            debug_dir = self.output_dir / 'debug'
            debug_dir.mkdir(exist_ok=True)
            
            debug_frame = frame.copy()
            x, y = detection_info['location']
            w, h = detection_info['size']
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(str(debug_dir / 'logo_detection.png'), debug_frame)
        
        self.metrics['logo_detection'] = detection_info
        return detection_info

    def replace_logo(self, frame_path, logo_info):
        """Replace logo using FFmpeg with CUDA acceleration."""
        output_path = self.processed_dir / frame_path.name
        x, y = logo_info['location']
        w, h = logo_info['size']
        
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',
            '-i', str(frame_path),
            '-i', str(self.new_logo),
            '-filter_complex',
            f'[1]scale={w}:{h}[logo];[0][logo]overlay={x}:{y}:format=auto',
            '-c:v', 'png',
            '-frames:v', '1',
            str(output_path)
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"\nFFmpeg Error for frame {frame_path.name}:")
            print(stderr.decode())
            return False
            
        return True

    def process_frame_batch(self, frame_batch, logo_info):
        """Process a batch of frames in parallel using GPU."""
        results = []
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            futures = [
                executor.submit(self.replace_logo, frame_path, logo_info)
                for frame_path in frame_batch
            ]
            for future in futures:
                results.append(future.result())
        return results

    def compile_video(self):
        """Compile video using GPU acceleration."""
        temp_video = self.output_dir / 'temp_output.mp4'
        input_name = Path(self.input_video).stem
        output_path = self.output_dir / f'{input_name}_full-v5_output.mp4'

        # Compile frames using GPU
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',
            '-framerate', str(self.metrics['video_metrics']['fps']),
            '-i', str(self.processed_dir / 'frame_%06d.png'),
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',
            '-rc', 'constqp',
            '-qp', '17',
            str(temp_video)
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise Exception(f"FFmpeg error during video compilation: {stderr.decode()}")

        # Rename temp video to final output
        temp_video.rename(output_path)

        compilation_metrics = {
            'output_size_mb': output_path.stat().st_size / (1024 * 1024),
            'compression_ratio': os.path.getsize(self.input_video) / output_path.stat().st_size
        }

        self.metrics['video_compilation'] = compilation_metrics
        return compilation_metrics

    def extract_frames(self):
        """Extract frames using GPU acceleration."""
        start_time = time.time()
        
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',
            '-i', str(self.input_video),
            '-vf', f"fps={self.metrics['video_metrics']['fps']}",
            '-qscale:v', '1',
            str(self.frames_dir / 'frame_%06d.png')
        ]

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
                    pbar.update(min(100, self.metrics['video_metrics']['frame_count'] - pbar.n))

        stdout, stderr = process.communicate()

        # Log the full error message for debugging purposes
        print("\nFFmpeg GPU Error Output:")
        print(stderr)

        if process.returncode != 0:
            raise Exception(f"GPU frame extraction failed with return code {process.returncode}")

        extracted_frames = list(self.frames_dir.glob('*.png'))
        extraction_time = time.time() - start_time

        frame_metrics = {
            'extraction_time': extraction_time,
            'frames_extracted': len(extracted_frames),
            'avg_time_per_frame': extraction_time / len(extracted_frames) if extracted_frames else 0
        }

        self.metrics['frame_extraction'] = frame_metrics
        return len(extracted_frames)

    def process_video(self):
        """Main processing pipeline with GPU acceleration and resource monitoring."""
        print("\n" + "="*80)
        print("GPU-ACCELERATED VIDEO PROCESSING PIPELINE - DETAILED MONITORING")
        print("="*80)
        print(f"Using {self.cpu_count} CPU cores and NVIDIA GPU for processing")
        
        overall_start = time.time()
        
        # Start resource monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_resources)
        monitor_thread.start()
        
        try:
            # Phase 1: Video Analysis
            print("\nPHASE 1: VIDEO ANALYSIS")
            print("="*50)
            video_info = self.get_video_info()
            print(f"\nVideo Analysis Results:")
            print(f"├── Resolution: {video_info['width']}x{video_info['height']}")
            print(f"├── FPS: {video_info['fps']:.2f}")
            print(f"├── Frame Count: {video_info['frame_count']}")
            print(f"├── Duration: {video_info['duration']:.2f}s")
            print(f"└── File Size: {video_info['file_size_mb']:.2f}MB")
            self.save_checkpoint('video_analysis')
            
            # Phase 2: Frame Extraction
            print("\nPHASE 2: GPU-ACCELERATED FRAME EXTRACTION")
            print("="*50)
            print(f"Using NVIDIA GPU acceleration for frame extraction")
            frames_count = self.extract_frames()
            print(f"\nFrame Extraction Results:")
            print(f"├── Frames Extracted: {frames_count}")
            print(f"├── Extraction Time: {self.metrics['frame_extraction']['extraction_time']:.2f}s")
            print(f"└── Average Time per Frame: {self.metrics['frame_extraction']['avg_time_per_frame']*1000:.2f}ms")
            self.save_checkpoint('frame_extraction')
            
            # Phase 3: Logo Detection and Replacement
            print("\nPHASE 3: LOGO DETECTION AND REPLACEMENT")
            print("="*50)
            
            frame_paths = sorted(self.frames_dir.glob('*.png'))
            if not frame_paths:
                raise Exception("No frames were extracted. Check frame extraction phase.")
            
            # Detect logo in first frame
            first_frame = frame_paths[0]
            logo_info = self.detect_initial_logo(first_frame)
            
            if not logo_info['success']:
                raise Exception(f"Logo detection failed. Confidence {logo_info['confidence']:.4f} below threshold")
            
            # Process frames in parallel batches
            batch_size = max(len(frame_paths) // (self.cpu_count * 4), 1)
            frame_batches = [frame_paths[i:i + batch_size]
                           for i in range(0, len(frame_paths), batch_size)]
            
            processed_count = 0
            success_count = 0
            
            print(f"\nProcessing Configuration:")
            print(f"├── Batch Size: {batch_size}")
            print(f"├── Total Batches: {len(frame_batches)}")
            print(f"└── Total Frames: {len(frame_paths)}")
            
            with tqdm(total=len(frame_paths), desc="Processing frames") as pbar:
                for batch_idx, batch in enumerate(frame_batches):
                    results = self.process_frame_batch(batch, logo_info)
                    processed_count += len(batch)
                    success_count += sum(1 for r in results if r)
                    
                    pbar.update(len(batch))
                    pbar.set_postfix({
                        'CPU': f"{psutil.cpu_percent()}%",
                        'Memory': f"{psutil.Process().memory_info().rss/1024/1024:.1f}MB",
                        'Success': f"{success_count}/{processed_count}"
                    })
                    
                    if processed_count % (batch_size * 10) == 0:
                        print(f"\nBatch {batch_idx + 1}/{len(frame_batches)} Complete:")
                        print(f"├── Frames Processed: {processed_count}/{len(frame_paths)}")
                        print(f"├── Success Rate: {(success_count/processed_count*100):.1f}%")
                        print(f"├── CPU Usage: {psutil.cpu_percent()}%")
                        print(f"└── Memory Usage: {psutil.Process().memory_info().rss/1024/1024:.1f}MB")
                        self.save_checkpoint('frame_processing', processed_count)
            
            print(f"\nLogo Replacement Complete:")
            print(f"├── Total Frames Processed: {processed_count}")
            print(f"├── Successful Replacements: {success_count}")
            print(f"├── Success Rate: {(success_count/processed_count*100):.1f}%")
            print(f"└── Failed Frames: {processed_count - success_count}")
            
            self.save_checkpoint('logo_processing')
            
            # Phase 4: Video Compilation
            print("\nPHASE 4: GPU-ACCELERATED VIDEO COMPILATION")
            print("="*50)
            print("Using NVIDIA NVENC for hardware-accelerated encoding")
            
            compilation_metrics = self.compile_video()
            print(f"\nVideo Compilation Results:")
            print(f"├── Output Size: {compilation_metrics['output_size_mb']:.2f}MB")
            print(f"└── Compression Ratio: {compilation_metrics['compression_ratio']:.2f}x")
            
            self.save_checkpoint('video_compilation')
            
            # Stop resource monitoring
            self.stop_monitoring.set()
            monitor_thread.join()
            
            # Final Summary
            total_time = time.time() - overall_start
            self.metrics['overall_metrics'].update({
                'total_time': total_time,
                'frames_per_second': frames_count/total_time,
                'peak_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'final_cpu_percent': psutil.cpu_percent(),
                'success_rate': success_count/processed_count,
                'resource_monitoring': self.resource_metrics
            })
            
            print("\nFINAL PROCESSING SUMMARY")
            print("="*50)
            print(f"Overall Performance:")
            print(f"├── Total Processing Time: {total_time:.2f}s")
            print(f"├── Processing Speed: {frames_count/total_time:.2f} frames/second")
            print(f"├── Peak Memory Usage: {self.metrics['overall_metrics']['peak_memory_mb']:.1f}MB")
            print(f"├── Final CPU Usage: {self.metrics['overall_metrics']['final_cpu_percent']}%")
            print(f"├── Frames Processed: {processed_count}")
            print(f"├── Success Rate: {(success_count/processed_count*100):.1f}%")
            print(f"└── GPU Acceleration: Active")
            
            # Save final metrics
            with self.metrics_lock:
                with open(self.metrics_file, 'w') as f:
                    json.dump(self.metrics, f, indent=4)
            
            self.update_stats("Processing Complete")
            return True
            
        except Exception as e:
            self.stop_monitoring.set()
            monitor_thread.join()
            print(f"\n❌ Error: {str(e)}")
            print(traceback.format_exc())
            self.update_stats(f"Error occurred: {str(e)}")
            raise

    def save_checkpoint(self, phase, frame_number=None):
        """Save processing checkpoint."""
        checkpoint_data = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'frame_number': frame_number,
            'metrics': self.metrics
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{phase}.json"
        with self.metrics_lock:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=4)
                
            self.metrics['checkpoints']['last_successful_phase'] = phase
            if frame_number:
                self.metrics['checkpoints']['frames_processed'] = frame_number

def main():
    processor = GPUVideoLogoReplacer(
        input_video=r"C:\Users\prady\Documents\Capstone\frame-manipulator\UE18CS315_2020-09-11_CLASS25_SJ_1min.mp4",
        old_logo=r"C:\Users\prady\Documents\Capstone\frame-manipulator\old-pes-logo.png",
        new_logo=r"C:\Users\prady\Documents\Capstone\frame-manipulator\new-pes-logo.png",
        output_dir=r"C:\Users\prady\Documents\Capstone\frame-manipulator"
    )
    processor.process_video()

if __name__ == "__main__":
    main()
