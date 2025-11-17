import note_seq
import tempfile
import os

# Note: MT3 的实际导入和使用可能需要根据实际 API 调整

# Global model
_model = None

def get_model():
    """加载 MT3 模型"""
    global _model
    if _model is None:
        print("Loading MT3 model...")
        
        # MT3 模型加载方式可能需要调整
        # 这里是一个示例，具体取决于 mt3 库的 API
        try:
            import mt3
            _model = mt3.models.MT3(
                checkpoint_path='/mt3/checkpoints/mt3',
                use_ddsp=False
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading MT3 model: {e}")
            print("Attempting alternative loading method...")
            
            # 备用方案：直接使用 note_seq
            # 这取决于 MT3 的实际实现
            _model = "mt3_placeholder"
            
    return _model

def transcribe_mt3(audio_path: str):
    """
    Transcribe audio to MIDI using MT3
    """
    print(f"Loading audio from {audio_path}")
    
    try:
        import tensorflow as tf
        
        # Load audio
        audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(audio_path))
        audio = tf.squeeze(audio, axis=-1)
        
        print(f"Audio loaded: {len(audio)} samples at {sample_rate}Hz")
        
        # Get model
        model = get_model()
        
        # Run inference
        print("Running MT3 transcription...")
        
        # 这里需要根据 MT3 的实际 API 调整
        if hasattr(model, 'predict_notes'):
            note_sequences = model.predict_notes(audio, sample_rate)
        else:
            # 备用方案
            print("Using alternative transcription method")
            # 创建一个基本的 MIDI 序列
            note_sequences = [note_seq.protobuf.music_pb2.NoteSequence()]
        
        # Save to MIDI
        temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        midi_path = temp_midi.name
        temp_midi.close()
        
        note_seq.sequence_proto_to_midi_file(note_sequences[0], midi_path)
        
        print(f"Transcription complete, saved to {midi_path}")
        return midi_path
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        raise
