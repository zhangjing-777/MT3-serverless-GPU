import runpod
import base64
import tempfile
import os
from predict import transcribe_mt3

def handler(event):
    """
    RunPod Serverless Entry
    Expected input:
    {
        "input": {
            "audio_base64": "<base64 audio>"
        }
    }
    """
    try:
        print("Received request")
        audio_b64 = event["input"]["audio_base64"]
        audio_bytes = base64.b64decode(audio_b64)
        
        print(f"Decoded audio: {len(audio_bytes)} bytes")

        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            audio_path = f.name
        
        print(f"Saved audio to {audio_path}")

        # Transcribe
        midi_path = transcribe_mt3(audio_path)

        # Convert MIDI to base64
        with open(midi_path, "rb") as f:
            midi_b64 = base64.b64encode(f.read()).decode()
        
        # Cleanup
        os.unlink(audio_path)
        os.unlink(midi_path)
        
        print("Request completed successfully")
        return {"midi_base64": midi_b64}
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
