import os
import json
import base64
import queue
import threading
import traceback
import time
import gc
from typing import Any, Dict, List
from dataclasses import dataclass

# Gemini / Vertex AI imports
import vertexai
from vertexai.generative_models import GenerativeModel, Part


@dataclass
class VideoEntry:
    mp4_path: str
    # optional keys below:
    youtube_key_segment: str = None
    duration: float = None
    fps: float = None
    height: int = None
    width: int = None
    n_frames: int = None
    # Add other metadata fields as needed


@dataclass
class CaptionResult:
    mp4_path: str
    caption: str
    # optional keys below:
    youtube_key_segment: str = None
    duration: float = None
    fps: float = None
    height: int = None
    width: int = None
    n_frames: int = None


class GeminiCaptionProcessor:
    def __init__(self, output_file: str, num_workers: int = 12):
        self.output_file = output_file
        self.num_workers = num_workers
        self.entry_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.workers = []
        self.success_count = 0
        self.fail_count = 0
        self.start_time = None
        self.end_time = None

        # Initialize Vertex AI
        PROJECT_ID = "<your project id>"
        model_index = 0
        LOCATION = ["us-central1", "us-east5"][model_index]
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        MODEL_NAME = ["gemini-2.0-flash-001"][model_index]  # "gemini-2.0-flash-001"
        self.model = GenerativeModel(model_name=MODEL_NAME)
        print(f"Using model: {MODEL_NAME}")

        self.prompt = (
            "Summarize this video directly, when summarizing please provide a detailed description of major subjects, actions, and interactions. "
            "Focus on key actions, interactions, and movements. Include camera movements. "
            "Keep the summary brief and clear. "
            "Only include information that is certain, and avoid speculation or assumptions."
            "In the last sentence, answer the question with just Yes or No, does the video contain rich human hand motions?"
        )
        # Lock for updating success and fail counts
        self.count_lock = threading.Lock()

        self.optional_keys = [
            "duration",
            "fps",
            "height",
            "width",
            "n_frames",
            "youtube_key_segment",
        ]

    def process_entries(self, records: List[Dict[str, Any]]):
        self.start_time = time.time()
        # Start worker threads
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_process, daemon=True)
            worker.start()
            self.workers.append(worker)

        # Producer: read input lines and put them into the queue
        to_process_count = 0
        for data in records:
            entry = VideoEntry(
                mp4_path=data["video_path"],
            )
            # add optional keys to entry:
            for key in self.optional_keys:
                if key in data:
                    entry.__dict__[key] = data[key]
            self.entry_queue.put(entry)
            to_process_count += 1

        if to_process_count == 0:
            print("No new entries to process. All done!")
            # Even if none, still send sentinels to avoid blocking
            for _ in range(self.num_workers):
                self.entry_queue.put(None)
            return

        # Add sentinel values to signal workers to stop
        for _ in range(self.num_workers):
            self.entry_queue.put(None)

        # Wait for all workers to finish
        for worker in self.workers:
            worker.join()

        # Collect results
        results = []
        while not self.results_queue.empty():
            result = self.results_queue.get()
            # Only append results that aren't error messages
            if not result.caption.startswith("Error"):
                results.append(result)

        # Append results to output file
        with open(self.output_file, "a", encoding="utf-8") as f:
            for result in results:
                obj = {"video_path": result.mp4_path, "caption": result.caption}
                for key in self.optional_keys:
                    if key in result.__dict__ and result.__dict__[key] is not None:
                        obj[key] = result.__dict__[key]
                f.write(json.dumps(obj) + "\n")

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        print(f"Processed {len(results)} entries successfully.")
        print(f"Failed on {self.fail_count} entries.")
        print(f"Total time: {total_time:.2f} seconds.")
        if to_process_count > 0:
            print(f"Throughput: {to_process_count / total_time:.2f} videos/second.")
        print(f"Output file: {self.output_file}")

    def _read_video_file(self, file_path):
        """Read video file and convert it to base64."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        with open(file_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")

    def get_gemini_caption(self, video_path: str) -> str:
        """Generate a caption for a single video using Gemini Flash."""
        video_data = self._read_video_file(video_path)
        video_part = Part.from_data(data=video_data, mime_type="video/mp4")
        try:
            response = self.model.generate_content(
                [video_part, self.prompt],
                # generation_config={
                #     "max_output_tokens": 1024,
                #     "temperature": 0.4
                # },
                stream=False,
            )
            return response.text
        except Exception as e:
            print(f"Error from Gemini API: {e}")
            return f"Error from Gemini API: {e}"

    def _process_single_entry(self, entry: VideoEntry) -> CaptionResult:
        caption = self.get_gemini_caption(entry.mp4_path)

        ret_result = CaptionResult(mp4_path=entry.mp4_path, caption=caption)
        for key in self.optional_keys:
            if key in entry.__dict__ and entry.__dict__[key] is not None:
                ret_result.__dict__[key] = entry.__dict__[key]
        return ret_result

    def _worker_process(self):
        while True:
            entry = self.entry_queue.get()
            if entry is None:  # Check for sentinel value
                break
            if self.entry_queue.qsize() % 100 == 0:
                print(
                    f"Processing {entry.mp4_path}. {self.entry_queue.qsize()} entries left in queue."
                )
                gc_s_time = time.time()
                num_gc = gc.collect()
                gc_e_time = time.time()
                print(
                    f"Garbage collection took {gc_e_time - gc_s_time} seconds, collected {num_gc} objects"
                )
            try:
                result = self._process_single_entry(entry)
                # Check if result is error. If not, add to results_queue.
                if not result.caption.startswith("Error"):
                    with self.count_lock:
                        self.success_count += 1
                    self.results_queue.put(result)
                else:
                    with self.count_lock:
                        self.fail_count += 1
                    print(f"Skipping {entry.mp4_path} due to error in captioning.")
            except Exception as e:
                with self.count_lock:
                    self.fail_count += 1
                print(f"Error processing {entry.mp4_path}: {str(e)}")
                traceback.print_exc()
            finally:
                self.entry_queue.task_done()
