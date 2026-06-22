import os
import json
import shutil
import time
import psutil
from tqdm import tqdm
import subprocess
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import logging

class VideoLogoReplacer:
    def __init__(self, input_video, old_logo, new_logo, output_dir="output"):
        """Main processing pipeline with comprehensive metrics tracking."""
        print("\n" + "="*80)
        print("VIDEO PROCESSING PIPELINE - DETAILED MONITORING")
        print("="*80)
        
        overall_start = time.time()
        process = psutil.Process()
        
        self.input_video = input_video
        self.old_logo = old_logo
        self.new_logo = new_logo
        self.output_dir = Path(output_dir)
        
        # Update directory names to include version
        self.frames_dir = self.output_dir / 'frames_full-v2'
        self.processed_dir = self.output_dir / 'processed_frames_full-v2'
        self.checkpoint_dir = self.output_dir / 'checkpoints_full-v2'
        self.metrics_file = self.output_dir / 'full-v2_metrics.json'
        
        # Create directories
        for dir_path in [self.output_dir, self.frames_dir, self.processed_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'video_metrics': self.get_video_info(),
            'checkpoints': {
                'last_successful_phase': None,
                'frames_processed': 0
            },
            'overall_metrics': {
                'start_time': datetime.now().isoformat(),
                'cpu_usage': psutil.cpu_percent(),
                'initial_memory': psutil.Process().memory_info().rss / 1024 / 1024
            }
        }
        
        self.logger = None
        
        self.success = False
        
        self.update_stats_file("Processing Started")
        
        try:
            # Phase 1: Video Analysis (same as full-v0)
            print("\nPHASE 1: VIDEO ANALYSIS")
            print("="*50)
            with tqdm(total=1, desc="Analyzing video") as pbar:
                video_info = self.get_video_info()
                print(f"\nVideo Analysis Results:")
                print(f"├── Resolution: {video_info['width']}x{video_info['height']}")
                print(f"├── FPS: {video_info['fps']:.2f}")
                print(f"├── Frame Count: {video_info['frame_count']}")
                print(f"└── Duration: {video_info['duration']:.2f}s")
                pbar.update(1)
            self.save_checkpoint('video_analysis')
            
            # Phase 2: Frame Extraction (same as full-v0)
            print("\nPHASE 2: FRAME EXTRACTION")
            print("="*50)
            total_frames = self.metrics['video_metrics']['frame_count']
            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                frames_count = self.extract_frames()
                pbar.update(total_frames)
            self.save_checkpoint('frame_extraction')
            
            # Phase 3: Frame Processing (MODIFIED)
            print("\nPHASE 3: FRAME PROCESSING")
            print("="*50)
            frame_paths = sorted(self.frames_dir.glob('*.png'))
            processed_count = 0
            logo_detected_count = 0
            
            # First, detect logo in the first frame
            first_frame = frame_paths[0]
            first_frame_logo = self.detect_logo_in_frame(first_frame)
            
            if not first_frame_logo['detected']:
                raise Exception("Could not detect logo in the first frame. Aborting process.")
            
            print(f"\nFirst Frame Logo Detection:")
            print(f"├── Confidence: {first_frame_logo['confidence']:.4f}")
            print(f"├── Location: {first_frame_logo['location']}")
            print(f"├── Size: {first_frame_logo['size']}")
            print(f"└── Detection Time: {first_frame_logo['detection_time']*1000:.2f}ms")
            
            # Store logo information for reuse
            logo_info = {
                'detected': True,
                'confidence': first_frame_logo['confidence'],
                'location': first_frame_logo['location'],
                'size': first_frame_logo['size']
            }
            
            # Process all frames using the same logo coordinates
            with tqdm(total=len(frame_paths), desc="Processing frames") as pbar:
                for frame_num, frame_path in enumerate(frame_paths, 1):
                    try:
                        # Use the same logo info for all frames
                        processed = self.replace_logo(frame_path, logo_info)
                        
                        processed_count += 1
                        if processed:
                            logo_detected_count += 1
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'CPU': f"{psutil.cpu_percent()}%",
                            'Memory': f"{process.memory_info().rss/1024/1024:.1f}MB",
                            'Frames': f"{processed_count}/{len(frame_paths)}"
                        })
                        
                        # Save checkpoint every 100 frames
                        if frame_num % 100 == 0:
                            self.save_checkpoint('frame_processing', frame_num)
                        
                    except Exception as e:
                        print(f"\n❌ Error processing frame {frame_num}: {str(e)}")
                        self.logger.error(f"Frame {frame_num} processing failed: {str(e)}")
                        self.save_checkpoint('frame_processing_failed', frame_num)
                        continue
            
            print(f"\nFrame Processing Summary:")
            print(f"├── Total Frames Processed: {processed_count}")
            print(f"├── Logos Replaced: {logo_detected_count}")
            print(f"├── Processed Frames Saved: {len(list(self.processed_dir.glob('*.png')))}")
            print(f"└── Processing Rate: {(processed_count/len(frame_paths)*100):.1f}%\n")
            
            # Phase 4: Video Compilation (same as full-v0)
            print("\nPHASE 4: VIDEO COMPILATION")
            print("="*50)
            with tqdm(total=1, desc="Compiling video") as pbar:
                try:
                    compilation_metrics = self.compile_video()
                    pbar.update(1)
                    self.save_checkpoint('video_compilation')
                except Exception as e:
                    print(f"\n❌ Video compilation failed: {str(e)}")
                    print("Saving processed frames and intermediate results...")
                    raise
            
            # Final Summary (same as full-v0)
            print("\nFINAL PROCESSING SUMMARY")
            print("="*50)
            total_time = time.time() - overall_start
            print(f"Overall Metrics:")
            print(f"├── Total Processing Time: {total_time:.2f} seconds")
            print(f"├── Average Processing Speed: {frames_count/total_time:.2f} frames/second")
            print(f"├── Peak Memory Usage: {process.memory_info().rss/1024/1024:.1f} MB")
            print(f"└── Success Rate: {(processed_count/len(frame_paths)*100):.1f}%\n")
            
            return True
            
        except Exception as e:
            print("\n" + "="*50)
            print("❌ ERROR OCCURRED - SAVING PROGRESS")
            print("="*50)
            print(f"Error: {str(e)}")
            print("\nCheckpoint Information:")
            print(f"├── Last Successful Phase: {self.metrics['checkpoints']['last_successful_phase']}")
            print(f"├── Frames Processed: {self.metrics['checkpoints']['frames_processed']}")
            print(f"└── Progress Saved In: {self.checkpoint_dir}")
            
            self.save_checkpoint('error_occurred')
            raise
        
        finally:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            if hasattr(self, 'success') and self.success:
                if os.path.exists(self.frames_dir):
                    shutil.rmtree(self.frames_dir)

    def extract_audio(self):
        """Extract audio from input video."""
        print("\nExtracting audio...")
        audio_path = self.output_dir / 'temp_audio.aac'
        
        cmd = [
            'ffmpeg', '-i', self.input_video,
            '-vn', '-acodec', 'copy',
            str(audio_path)
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print("⚠️ No audio found in video or extraction failed")
            return None
        
        return audio_path

    def compile_video(self):
        """Compile processed frames back into video with original audio."""
        start_time = time.time()
        temp_video = self.output_dir / 'temp_output.mp4'
        input_name = Path(self.input_video).stem
        output_path = self.output_dir / f'{input_name}_full-v2_output.mp4'
        
        try:
            # Extract audio first
            audio_path = self.extract_audio()
            
            # Compile frames into temporary video
            cmd = [
                'ffmpeg', '-framerate', str(self.metrics['video_metrics']['fps']),
                '-i', str(self.processed_dir / 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                str(temp_video)
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"FFmpeg error during video compilation: {stderr.decode()}")
            
            # Add audio back if it exists
            if audio_path and audio_path.exists():
                cmd = [
                    'ffmpeg', '-i', str(temp_video),
                    '-i', str(audio_path),
                    '-c:v', 'copy', '-c:a', 'aac',
                    str(output_path)
                ]
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                # Clean up temporary files
                temp_video.unlink()
                audio_path.unlink()
            else:
                # If no audio, just rename temp video
                temp_video.rename(output_path)
            
            # Record metrics
            self.metrics['video_compilation'] = {
                'compilation_time': time.time() - start_time,
                'output_size': output_path.stat().st_size,
                'compression_ratio': os.path.getsize(self.input_video) / output_path.stat().st_size,
                'has_audio': audio_path is not None
            }
            
            return self.metrics['video_compilation']
            
        except Exception as e:
            if temp_video.exists():
                temp_video.unlink()
            if audio_path and audio_path.exists():
                audio_path.unlink()
            raise

    def update_stats_file(self, message):
        """Update the stats file with new information."""
        stats_file = self.output_dir / 'full-v2_stats.txt'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(stats_file, 'a') as f:
            f.write(f"\n[{timestamp}] {message}")
            
            # Add current system metrics
            f.write(f"\nCPU Usage: {psutil.cpu_percent()}%")
            f.write(f"\nMemory Usage: {psutil.Process().memory_info().rss/1024/1024:.1f}MB")
            f.write("\n" + "="*50 + "\n")

    def get_video_info(self):
        """Extract video information and store metrics."""
        cap = cv2.VideoCapture(self.input_video)
        video_metrics = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return video_metrics

    def extract_frames(self):
        """Extract frames using ffmpeg with performance tracking."""
        start_time = time.time()
        
        # Construct ffmpeg command
        cmd = [
            'ffmpeg', '-i', self.input_video,
            '-vf', 'fps=' + str(self.metrics['video_metrics']['fps']),
            str(self.frames_dir / 'frame_%06d.png')
        ]
        
        # Execute and measure
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Record metrics
        self.metrics['frame_extraction'] = {
            'extraction_time': time.time() - start_time,
            'frames_extracted': len(list(self.frames_dir.glob('*.png'))),
            'avg_time_per_frame': (time.time() - start_time) / self.metrics['video_metrics']['frame_count']
        }
        
        return self.metrics['frame_extraction']['frames_extracted']

    def detect_logo_in_frame(self, frame_path):
        """Detect logo in a single frame using multi-scale template matching."""
        frame = cv2.imread(str(frame_path))
        template = cv2.imread(self.old_logo)
        
        detection_start = time.time()
        best_confidence = 0
        best_loc = None
        best_size = None
        
        # Try multiple scales from 0.8x to 2.0x
        scales = np.linspace(0.8, 2.0, 25)
        
        for scale in scales:
            # Resize template according to scale
            width = int(template.shape[1] * scale)
            height = int(template.shape[0] * scale)
            resized_template = cv2.resize(template, (width, height))
            
            # Template matching
            result = cv2.matchTemplate(frame, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Update best match if current is better
            if max_val > best_confidence:
                best_confidence = max_val
                best_loc = max_loc
                best_size = (height, width)
        
        detection_time = time.time() - detection_start
        
        # Record detection metrics for this frame
        frame_metrics = {
            'detected': best_confidence > 0.3,
            'confidence': float(best_confidence),
            'location': best_loc if best_confidence > 0.3 else None,
            'size': best_size if best_confidence > 0.3 else None,
            'detection_time': detection_time,
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_usage': psutil.cpu_percent()
        }
        
        return frame_metrics

    def replace_logo(self, frame_path, logo_info):
        """Replace detected logo with new logo or copy original frame if no logo detected."""
        frame = cv2.imread(str(frame_path))
        output_path = self.processed_dir / frame_path.name
        
        if not logo_info['detected']:
            cv2.imwrite(str(output_path), frame)
            return False
        
        replacement_start = time.time()
        
        # Load and resize new logo
        new_logo = cv2.imread(self.new_logo, cv2.IMREAD_UNCHANGED)
        h, w = logo_info['size']
        new_logo = cv2.resize(new_logo, (w, h))
        
        # Create mask for blending
        if new_logo.shape[2] == 4:  # If PNG with alpha channel
            alpha = new_logo[:, :, 3] / 255.0
            alpha = cv2.merge([alpha, alpha, alpha])
            new_logo = new_logo[:, :, :3]
        else:
            alpha = np.ones((h, w, 3))
        
        # Get region of interest with boundary checks
        x, y = logo_info['location']
        frame_h, frame_w = frame.shape[:2]
        
        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y
        
        roi = frame[y:y+h, x:x+w]
        
        # Resize components if needed
        if roi.shape != new_logo[:h, :w].shape:
            new_logo = cv2.resize(new_logo, (w, h))
            alpha = cv2.resize(alpha, (w, h))
        
        # Blend logos
        blended = (1 - alpha[:h, :w]) * roi + alpha[:h, :w] * new_logo[:h, :w]
        frame[y:y+h, x:x+w] = blended
        
        # Save processed frame
        cv2.imwrite(str(output_path), frame)
        return True

    def save_checkpoint(self, phase, frame_number=None):
        """Save processing checkpoint."""
        checkpoint_data = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'frame_number': frame_number,
            'metrics': self.metrics
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{phase}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
            
        self.metrics['checkpoints']['last_successful_phase'] = phase
        if frame_number:
            self.metrics['checkpoints']['frames_processed'] = frame_number

def main():
    # Same as full-v0
    processor = VideoLogoReplacer(
        input_video=r"C:\Users\prady\Documents\Capstone\frame-manipulator\UE18CS315_2020-09-11_CLASS25_SJ_1min.mp4",
        old_logo=r"C:\Users\prady\Documents\Capstone\frame-manipulator\old-pes-logo.png",
        new_logo=r"C:\Users\prady\Documents\Capstone\frame-manipulator\new-pes-logo.png",
        output_dir=r"C:\Users\prady\Documents\Capstone\frame-manipulator"
    )
    processor.process_video()

if __name__ == "__main__":
    main()
