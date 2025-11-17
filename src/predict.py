import mt3
import note_seq
import tensorflow as tf
import tempfile

# Global model
_model = None

def get_model():
    """加载 MT3 模型"""
    global _model
    if _model is None:
        print("Loading MT3 model...")
        _model = mt3.models.MT3(
            checkpoint_path='/mt3/checkpoints/mt3',
            use_ddsp=False
        )
        print("Model loaded successfully!")
    return _model

def transcribe_mt3(audio_path: str):
    """
    Transcribe audio to MIDI using MT3
    """
    print(f"Loading audio from {audio_path}")
    
    # Load audio
    audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(audio_path))
    audio = tf.squeeze(audio, axis=-1)
    
    print(f"Audio loaded: {len(audio)} samples at {sample_rate}Hz")
    
    # Get model
    model = get_model()
    
    # Run inference
    print("Running MT3 transcription...")
    note_sequences = model.predict_notes(audio, sample_rate)
    
    # Save to MIDI
    temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    midi_path = temp_midi.name
    temp_midi.close()
    
    note_seq.sequence_proto_to_midi_file(note_sequences[0], midi_path)
    
    print(f"Transcription complete, saved to {midi_path}")
    return midi_path
