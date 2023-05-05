import uuid
import logging
import math
import os
import tempfile
import zipfile
import time
import shutil
from multiprocessing import Pool

import gradio as gr
import jax.numpy as jnp
import numpy as np
import yt_dlp as youtube_dl
from jax.experimental.compilation_cache import compilation_cache as cc
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read

from whisper_jax import FlaxWhisperPipline


cc.initialize_cache("./jax_cache")
checkpoint = "openai/whisper-tiny"

DEBUG = False
BATCH_SIZE = 32
CHUNK_LENGTH_S = 30
NUM_PROC = 32
FILE_LIMIT_MB = 100000
YT_LENGTH_LIMIT_S = 72000  # limit to 2 hour YouTube files

title = description = article = " Whisper JAX ⚡️ "

language_names = sorted(TO_LANGUAGE_CODE.keys())

logger = logging.getLogger("whisper-jax-app")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

temp_path_zip_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')


def identity(batch):
    return batch


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    if seconds is None:
        # we have a malformed timestamp so just return it as is
        return seconds
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


if __name__ == "__main__":
    pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE)
    stride_length_s = CHUNK_LENGTH_S / 6
    chunk_len = round(CHUNK_LENGTH_S * pipeline.feature_extractor.sampling_rate)
    stride_left = stride_right = round(stride_length_s * pipeline.feature_extractor.sampling_rate)
    step = chunk_len - stride_left - stride_right
    pool = Pool(NUM_PROC)

    #do a pre-compile step so that the first user to use the demo isn't hit with a long transcription time
    logger.info("compiling forward call...")
    start = time.time()
    random_inputs = {"input_features": np.ones((BATCH_SIZE, 80, 3000))}
    random_timestamps = pipeline.forward(random_inputs, batch_size=BATCH_SIZE, return_timestamps=True)
    compile_time = time.time() - start
    logger.info(f"compiled in {compile_time}s")

    def create_transcript_zip(videos,tmpdir):
        """
      Clear the temporary directory contents
      
      Create a zip file for each video transcript and return the path to the zip of all transcripts.

      Args:
      videos (list of dict): Each dictionary must have "title" and "transcript" keys, containing the video title
      and its transcript respectively.

      Returns:
      str: Path to the zip file containing all transcript zip files.
      """
        for filename in os.listdir(tmpdir):
            file_path = os.path.join(tmpdir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f'Deleted {file_path}')
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        # Create a temporary directory to store all transcript zip files
        zip_paths = []
        # Loop through all videos and create a transcript zip file for each
        for video in videos:
            # Create a zip file with the video title as the filename
            zip_path = os.path.join(tmpdir, f"{video['title']}.zip")
            if not os.path.exists(temp_path_zip_file):
              os.makedirs(temp_path_zip_file)
            with zipfile.ZipFile(zip_path, "w") as zip_file:
                # Write the transcript to an SRT file with the same name as the video
                srt_path = os.path.join(tmpdir, f"{video['title']}.srt")
                with open(srt_path, "w") as srt_file:
                    srt_file.write(video["transcript"])
                # Add the SRT file to the zip
                zip_file.write(srt_path, f"{video['title']}.srt")
            zip_paths.append(zip_path)
        # Create a zip file containing all transcript zip files
        all_zip_path = os.path.join(tmpdir, "all_transcripts.zip")
        with zipfile.ZipFile(all_zip_path, "w") as all_zip_file:
            for zip_path in zip_paths:
                all_zip_file.write(zip_path, os.path.basename(zip_path))
        return all_zip_path

    def tqdm_generate(inputs: dict, task: str, return_timestamps: bool, progress: gr.Progress):
        inputs_len = inputs["array"].shape[0]
        all_chunk_start_idx = np.arange(0, inputs_len, step)
        num_samples = len(all_chunk_start_idx)
        num_batches = math.ceil(num_samples / BATCH_SIZE)
        dummy_batches = list(
            range(num_batches)
        )  # Gradio progress bar not compatible with generator, see https://github.com/gradio-app/gradio/issues/3841

        dataloader = pipeline.preprocess_batch(inputs, chunk_length_s=CHUNK_LENGTH_S, batch_size=BATCH_SIZE)
        progress(0, desc="Pre-processing audio file...")
        logger.info("pre-processing audio file...")
        dataloader = pool.map(identity, dataloader)
        logger.info("done post-processing")

        start_time = time.time()
        logger.info("transcribing...")
        model_outputs = [
            pipeline.forward(
                batch, batch_size=BATCH_SIZE, task=task, return_timestamps=True
            )
            for batch, _ in zip(
                dataloader, progress.tqdm(dummy_batches, desc="Transcribing...")
            )
        ]
        runtime = time.time() - start_time
        logger.info("done transcription")

        logger.info("post-processing...")
        post_processed = pipeline.postprocess(model_outputs, return_timestamps=True)
        text = post_processed["text"]
        if return_timestamps:
            timestamps = post_processed.get("chunks")
            timestamps = [
                f"[{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
                for chunk in timestamps
            ]
            text = "\n".join(str(feature) for feature in timestamps)
        logger.info("done post-processing")
        return text, runtime

    def transcribe_chunked_audio(inputs, task, return_timestamps, progress=gr.Progress()):
        progress(0, desc="Loading audio file...")
        logger.info("loading audio file...")
        if inputs is None:
            logger.warning("No audio file")
            raise gr.Error("No audio file submitted! Please upload an audio file before submitting your request.")
        file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
        if file_size_mb > FILE_LIMIT_MB:
            logger.warning("Max file size exceeded")
            raise gr.Error(
                f"File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB."
            )

        with open(inputs, "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
        logger.info("done loading")
        text, runtime = tqdm_generate(inputs, task=task, return_timestamps=return_timestamps, progress=progress)
        return text, runtime

    def _return_yt_html_embed(yt_url):
        video_id = yt_url[-1].split("?v=")[-1]
        return f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe> </center>'

    def download_yt_audio(yt_url, filename):
        title_ytb = youtube_dl.YoutubeDL().extract_info(yt_url, download=False).get("title", None)
        info_loader = youtube_dl.YoutubeDL()
        try:
            info = info_loader.extract_info(yt_url, download=False)
        except youtube_dl.utils.DownloadError as err:
            raise gr.Error(str(err)) from err

        file_length = info["duration_string"]
        file_h_m_s = file_length.split(":")
        file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
        if len(file_h_m_s) == 1:
            file_h_m_s.insert(0, 0)
        if len(file_h_m_s) == 2:
            file_h_m_s.insert(0, 0)

        file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]
        if file_length_s > YT_LENGTH_LIMIT_S:
            yt_length_limit_hms = time.strftime("%HH:%MM:%SS", time.gmtime(YT_LENGTH_LIMIT_S))
            file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
            raise gr.Error(f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video.")

        ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([yt_url])
                return title_ytb
            except youtube_dl.utils.ExtractorError as err:
                raise gr.Error(str(err))    

    def transcribe_youtube(yt_urls, task, return_timestamps, progress=gr.Progress()):
        final_files_data = []
        yt_urls = yt_urls.split()
        html_embed_str = _return_yt_html_embed(yt_urls)
        with tempfile.TemporaryDirectory() as tmpdirname:
            for yt_url in yt_urls:
                ran_id = str(uuid.uuid4())
                filepath = os.path.join(tmpdirname, f"{ran_id}_video.mp4")
                print(f"\n--Doing for {yt_urls.index(yt_url)}--{filepath}----\n")
                title_ytb = download_yt_audio(yt_url, filepath)
                with open(filepath, "rb") as f:
                    inputs = f.read()

                inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
                inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
                logger.info("done loading...")
                text, runtime = tqdm_generate(inputs, task=task, return_timestamps=return_timestamps, progress=progress)
                final_files_data.append({"title": title_ytb, "transcript": text})
#        print(final_files_data)
        path_of_zip_file = create_transcript_zip(final_files_data, temp_path_zip_file)
        return html_embed_str, path_of_zip_file, runtime

        
    microphone_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.inputs.Audio(source="microphone", optional=True, type="filepath"),
            gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
        ],
        allow_flagging="never",
        title=title,
        description=description,
        article=article,
    )

    audio_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.inputs.Audio(source="upload", optional=True, label="Audio file", type="filepath"),
            gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
        ],
        allow_flagging="never",
        title=title,
        description=description,
        article=article,
    )

    youtube = gr.Interface(
        fn=transcribe_youtube,
        inputs=[
            gr.inputs.Textbox(lines=5, placeholder="Paste the URLs of YouTube videos here, one per line", label="YouTube URLs"),
            gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.HTML(label="Video"),
            gr.outputs.File(label="Download files here"),
            gr.outputs.Textbox(label="Transcription Time (s)"),
        ],
        allow_flagging="never",
        title=title,
        cache_examples=False,
        description=description,
        article=article,
    )

    demo = gr.Blocks()

    with demo:
        gr.TabbedInterface([microphone_chunked, audio_chunked, youtube], ["Microphone", "Audio File", "YouTube"])

    demo.queue(concurrency_count=1, max_size=5)
    if DEBUG:
        demo.launch(server_name="0.0.0.0", show_api=False, share=True)
    else:
        demo.launch(server_name="0.0.0.0", show_api=False)
