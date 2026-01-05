#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Asynchronous image writing utilities for high-frequency data recording.

This module provides functionality for writing images to disk asynchronously
using multithreading or multiprocessing, which is critical for controlling
robots and recording data at high frame rates without blocking the main process.

The module supports two execution models:

    1. Threading mode (num_processes=0): Creates a pool of worker threads
       for concurrent image writing within a single process.
    2. Multiprocessing mode (num_processes>0): Creates multiple processes,
       each with their own thread pool, for maximum parallelism.

Key Features:
    - Asynchronous writing: Images are queued and written in background
      workers, preventing I/O blocking of the main process.
    - Multiple input formats: Supports torch Tensors, numpy arrays, and
      PIL Images with automatic conversion.
    - Format flexibility: Handles both channel-first (C, H, W) and
      channel-last (H, W, C) image formats.
    - Type conversion: Automatically converts float arrays in [0, 1] to
      uint8 in [0, 255] for PIL Image compatibility.
    - Safe cleanup: Decorator ensures image writers are properly stopped
      even when exceptions occur.

Classes:

    AsyncImageWriter
        Main class for asynchronous image writing with configurable threading
        or multiprocessing backends.

Functions:
    image_array_to_pil_image
        Convert numpy array to PIL Image with format and type conversion.
    write_image
        Write an image (numpy array or PIL Image) to disk.
    worker_thread_loop
        Worker thread loop for processing image write queue.
    worker_process
        Worker process that manages multiple threads for image writing.
    safe_stop_image_writer
        Decorator to safely stop image writer on exceptions.

Example:
    Create an async image writer with threading:
        >>> writer = AsyncImageWriter(num_processes=0, num_threads=4)
        >>> writer.save_image(image_array, Path("output/image.jpg"))
        >>> writer.wait_until_done()  # Wait for all images to be written
        >>> writer.stop()  # Clean up resources
"""

import multiprocessing
import queue
import threading
from pathlib import Path

import numpy as np
import PIL.Image
import torch


def safe_stop_image_writer(func):
    """Decorator to safely stop image writer on exceptions.

    Ensures that the image writer is properly stopped if an exception occurs
    during function execution.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that stops image writer on exceptions.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            dataset = kwargs.get("dataset")
            image_writer = getattr(dataset, "image_writer", None) if dataset else None
            if image_writer is not None:
                print("Waiting for image writer to terminate...")
                image_writer.stop()
            raise e

    return wrapper


def image_array_to_pil_image(image_array: np.ndarray, range_check: bool = True) -> PIL.Image.Image:
    """Convert a numpy array to a PIL Image.

    Supports channel-first (C, H, W) and channel-last (H, W, C) formats.
    Converts float arrays in [0, 1] to uint8 in [0, 255].

    Args:
        image_array: Input image array of shape (C, H, W) or (H, W, C).
        range_check: If True, validates that float arrays are in [0, 1] range.
            Defaults to True.

    Returns:
        PIL Image object.

    Raises:
        ValueError: If array has wrong number of dimensions, wrong number of
            channels, or float values are outside [0, 1] range.
        NotImplementedError: If image doesn't have 3 channels.

    Note:
        TODO(aliberts): handle 1 channel and 4 for depth images
    """
    # TODO(aliberts): handle 1 channel and 4 for depth images
    if image_array.ndim != 3:
        raise ValueError(f"The array has {image_array.ndim} dimensions, but 3 is expected for an image.")

    if image_array.shape[0] == 3:
        # Transpose from pytorch convention (C, H, W) to (H, W, C)
        image_array = image_array.transpose(1, 2, 0)

    elif image_array.shape[-1] != 3:
        raise NotImplementedError(
            f"The image has {image_array.shape[-1]} channels, but 3 is required for now."
        )

    if image_array.dtype != np.uint8:
        if range_check:
            max_ = image_array.max().item()
            min_ = image_array.min().item()
            if max_ > 1.0 or min_ < 0.0:
                raise ValueError(
                    "The image data type is float, which requires values in the range [0.0, 1.0]. "
                    f"However, the provided range is [{min_}, {max_}]. Please adjust the range or "
                    "provide a uint8 image with values in the range [0, 255]."
                )

        image_array = (image_array * 255).astype(np.uint8)

    return PIL.Image.fromarray(image_array)


def write_image(image: np.ndarray | PIL.Image.Image, fpath: Path) -> None:
    """Write an image to disk.

    Converts numpy arrays to PIL Images if needed, then saves to the specified path.

    Args:
        image: Image to save (numpy array or PIL Image).
        fpath: Path where the image will be saved.

    Raises:
        TypeError: If image type is not supported.
    """
    try:
        if isinstance(image, np.ndarray):
            img = image_array_to_pil_image(image)
        elif isinstance(image, PIL.Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        img.save(fpath)
    except Exception as e:
        print(f"Error writing image {fpath}: {e}")


def worker_thread_loop(queue: queue.Queue) -> None:
    """Worker thread loop for asynchronous image writing.

    Continuously processes items from the queue until receiving None (sentinel).
    Each item should be a tuple of (image_array, file_path).

    Args:
        queue: Queue containing (image_array, file_path) tuples or None sentinel.
    """
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        image_array, fpath = item
        write_image(image_array, fpath)
        queue.task_done()


def worker_process(queue: queue.Queue, num_threads: int) -> None:
    """Worker process that manages multiple threads for image writing.

    Creates and manages a pool of worker threads that process items from the queue.

    Args:
        queue: Queue containing (image_array, file_path) tuples or None sentinels.
        num_threads: Number of worker threads to create in this process.
    """
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread_loop, args=(queue,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class AsyncImageWriter:
    """
    This class abstract away the initialisation of processes or/and threads to
    save images on disk asynchrounously, which is critical to control a robot and record data
    at a high frame rate.

    When `num_processes=0`, it creates a threads pool of size `num_threads`.
    When `num_processes>0`, it creates processes pool of size `num_processes`, where each subprocess starts
    their own threads pool of size `num_threads`.

    The optimal number of processes and threads depends on your computer capabilities.
    We advise to use 4 threads per camera with 0 processes. If the fps is not stable, try to increase or lower
    the number of threads. If it is still not stable, try to use 1 subprocess, or more.
    """

    def __init__(self, num_processes: int = 0, num_threads: int = 1):
        self.num_processes = num_processes
        self.num_threads = num_threads
        self.queue = None
        self.threads = []
        self.processes = []
        self._stopped = False

        if num_threads <= 0 and num_processes <= 0:
            raise ValueError("Number of threads and processes must be greater than zero.")

        if self.num_processes == 0:
            # Use threading
            self.queue = queue.Queue()
            for _ in range(self.num_threads):
                t = threading.Thread(target=worker_thread_loop, args=(self.queue,))
                t.daemon = True
                t.start()
                self.threads.append(t)
        else:
            # Use multiprocessing
            self.queue = multiprocessing.JoinableQueue()
            for _ in range(self.num_processes):
                p = multiprocessing.Process(target=worker_process, args=(self.queue, self.num_threads))
                p.daemon = True
                p.start()
                self.processes.append(p)

    def save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path) -> None:
        """Queue an image for asynchronous writing.

        Converts torch tensors to numpy arrays and adds the image to the write queue.

        Args:
            image: Image to save (torch Tensor, numpy array, or PIL Image).
            fpath: Path where the image will be saved.
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy array to minimize main process time
            image = image.cpu().numpy()
        self.queue.put((image, fpath))

    def wait_until_done(self) -> None:
        """Wait until all queued images have been written to disk."""
        self.queue.join()

    def stop(self) -> None:
        """Stop all worker threads/processes and clean up resources.

        Sends sentinel values to all workers and waits for them to finish.
        Terminates processes if they don't respond.
        """
        if self._stopped:
            return

        if self.num_processes == 0:
            for _ in self.threads:
                self.queue.put(None)
            for t in self.threads:
                t.join()
        else:
            num_nones = self.num_processes * self.num_threads
            for _ in range(num_nones):
                self.queue.put(None)
            for p in self.processes:
                p.join()
                if p.is_alive():
                    p.terminate()
            self.queue.close()
            self.queue.join_thread()

        self._stopped = True
