# import necessary libraries
import asyncio, json, os
import sounddevice as sd # Capturing sound/cross platform audio & microphone capture
import webrtcvad # Voice activity detection. Tiny fast voice/activity detection
from faster_whisper import WhisperModel  # Audio to text processing. Local transcription - fast whisperer implementation
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import queue, threading, time, numpy as np # queue is used for audio handoff, threading for background worker, time for simple sleeps
# and numpy is used for arrays and audio buffers

# Audio pipeline settings
SAMPLE_RATE = 16000 # Whisper takes 16KHz mono audio input - compatible with VAD + models
CHANNELS = 1 # Mono audio input
FRAME_MS = 30 # Length of the audio for each sample i.e each_frame = (0.03) * 16000 samp/sec = 480 samples (input for VAD)
CHUNK_SECONDS = 1.5 # After VAD says speech/silent, chunk of audio is sent to model for conversion
OVERLAP_SECONDS = 0.3 # Bit of overlap to preserve the boundaries of words and don't miss them
VAD_AGGRESSIVENESS = 2 # Tuning of VAD on picking voice (0-3) 0 means picks very sensitive voices too but picks background noises also

# Load the model
model = WhisperModel("small", compute_type="int8") # "small" - model size (small for good and fast processing)

# Queueing and threading. Raw audio from mic -> sound device uses callback function, these threads are given for 
# processing -> processing_threads
audio_q = queue.Queue()

# Initialize VAD___
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) # creating a webrtc vad with specific aggressiveness
frame_len = int(SAMPLE_RATE * (FRAME_MS/1000.0)) # number of samples that can go per VAD frame (i.e 480 samples at 18 Khz per 30ms frame) 

# VAD can only take PCM (pulse code modulation) signals as input, so we have to change the dtype from float32 to int16 and specific range other than what
# sound device outputs i.e from [-1,1] to [-32768, 32768]

def to_pcm16(x):
    """
    Converting float32 to 16 bit pcm(int16) which vad can process
    """
    x = np.clip(x,-1,1) # avoid overflow if the clips exceeds between [-1,1] due to large spikes or something
    return (x * 32767).astype(np.int16) # scaling it to the range and converting the data type

def callback(indata, frames, t, status):
    """
    Microphone callback called by sounddevice:
    indata: pulls micblocks from the queue
    frames: number of samples in this block                             
    t: timing info
    status: xruns/overflows info 
    """
    if status:     # Just passing the audio, we can add else statement to log if there's any warning (dropout etc.)
        pass
    audio_q.put(indata.copy()) # pushing a copy to the queue


def vad_filtered_stream():
    """
    Important function - for slicing the audio, making into chunks and VAD usage to keep speech frames
    """    
    ring = np.zeros((0,CHANNELS),dtype=np.float32) # accumulating the audio (mic data) until we have full frame - ring buffer
    voiced = np.zeros(0,dtype=np.int16) #  buffer of PCM16 samples after VAD filtering
    while True: # Infinite generator, consumer controls the life time
        block = audio_q.get() # wait for next audio block from callback thread
        ring = np.vstack([ring, block]) # append new audio to the ring buffer
        
        # Process the ring buffer in fix sized VAD frames (FRAME_MS)
        while len(ring) >= frame_len: # while we have atleast one full length VAD frame
            frame = ring[:frame_len] # take the next frame (float32 mono chunk)
            ring = ring[frame_len:] # empty the ring buffer 

            pcm = to_pcm16(frame[:,0]) # convert the frame to mono int16 PCM (VAD requires bytes)
            is_speech = vad.is_speech(pcm.tobytes(),SAMPLE_RATE) # classify the frame

            if is_speech:
                voiced = np.concatenate([voiced, pcm]) # keep such frames contigous
            else:
                voiced = np.concatenate([voiced, np.zeros(frame_len, dtype=np.int16)]) # Insert small zero gaps to seperate each speech segments; helps whisper not glue words across silences
            
            # When we've accumulated enough chunk seconds of material ( speech + tiny gaps ) yeild a window for ASR
            target = int(SAMPLE_RATE * CHUNK_SECONDS)
            if len(voiced)>= target:
                # yeild the last window plus some overlap seconds to avoid cutting words
                yield voiced[-target - int(SAMPLE_RATE*OVERLAP_SECONDS):]
                # keep overlap tail so next window includes context but not the whole history
                keep =  int(SAMPLE_RATE * OVERLAP_SECONDS)
                voiced = voiced[-keep:]

#========== Background transcriber that pushes results into an asyncio queue =============
asyncio_queue: asyncio.Queue[str] = asyncio.Queue()


def transcribe_loop():
    """
    runs in background
    iterates over chunks of vad filtered stream
    converts them into float32 in [-1,1]
    calls fast whisperer model
    prints incremental text
    """
    print("Listening!!...... Press Ctrl+c to stop")
    for chunk in vad_filtered_stream(): # get rolling windows of ~1.2s (plus overlap)
        audio_f32 = (chunk.astype(np.float32)/32768.0) # convert back to float32 for whisper model
        segments, info = model.transcribe(
            audio_f32,
            language="en", # we can put none for auto detecting
            vad_filter=False, # we already did VAD, let whisper purely focus on transcription
            beam_size=1 # beam = 1 means greedy decoding, faster but slightly less accurate than larger beams
        )
        text = "".join([s.text for s in segments]).strip() # concatenate partial segment texts into one string
        if text:
            print(text)


def start_audio():
    """
    - Starts the background transcribe loop thread.
    - Opens a live input stream from default microphone
    - Keeps the main thread alive while the worker runs
    """
    t = threading.Thread(target=transcribe_loop, daemon=True) # daemon thread so process exits cleanly with ctrl+c
    t.start() # Start ASR worker
    return sd.InputStream( # Open mic stream
        channels=CHANNELS, 
        samplerate=SAMPLE_RATE,
        callback=callback,  # audio arrives here, goes into audio_q
        blocksize=frame_len # asking port audio to deliver ~FRAME_MS chunks
    )

# ==========FAST API with Websocket Endpoint========
app = FastAPI()
mic_stream_ctx = None

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type":"status","message":"connected"})
    try:
        while True:
            # Non blocking get with small timeout to allow pings/messages from client
            try:
                text = await asyncio.wait_for(asyncio_queue.get(), timeout=0.25)
                await ws.send_json({"type":"partial", "text":text})
            except asyncio.TimeoutError:
                # keep alive, also handle incoming control messages if you add them later
                pass
    except Exception:
        await ws.close()

@app.get("/")
def index():
    return HTMLResponse("<h3>STT server running. Connect add-in to ws://localhost:8000/ws/transcribe</h3>")

if __name__ == "__main__":
    # start audio + worker
    mic_stream_ctx = start_audio()
    mic_stream_ctx.__enter__() # Open stream
    try:
        CERT = os.path.expanduser("~/.office-addin-dev-certs/localhost.crt")
        KEY = os.path.expanduser("~/.office-addin-dev-certs/localhost.key")

        uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile=CERT, ssl_keyfile=KEY)
    finally:
        mic_stream_ctx.__exit__(None, None, None)
