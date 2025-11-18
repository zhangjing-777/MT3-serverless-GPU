import runpod
import sys

print("=" * 60, flush=True)
print("Handler starting...", flush=True)
print("Python version:", sys.version, flush=True)
print("=" * 60, flush=True)

# 测试导入
try:
    print("Testing imports...", flush=True)
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}", flush=True)
    
    import jax
    print(f"✓ JAX: {jax.__version__}", flush=True)
    
    import note_seq
    print("✓ note_seq imported", flush=True)
    
    print("✓ All imports successful!", flush=True)
except Exception as e:
    print(f"✗ Import failed: {e}", flush=True)
    import traceback
    traceback.print_exc()

def handler(event):
    """简化的测试 handler"""
    try:
        print("Received request", flush=True)
        return {"status": "ok", "message": "Handler is working"}
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting RunPod handler...", flush=True)
    runpod.serverless.start({"handler": handler})
