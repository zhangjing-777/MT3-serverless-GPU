import runpod
import base64
import tempfile
import os
import boto3
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from predict import transcribe_mt3

def upload_to_s3(file_path, bucket_name, object_key, expire_seconds=3600):
    """
    上传文件到 S3 并返回预签名 URL
    """
    try:
        # 从环境变量获取 AWS 凭证
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION", "ap-southeast-1")
        
        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS credentials not found in environment variables")
        
        # 创建 S3 客户端
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # 上传文件
        print(f"Uploading to S3: s3://{bucket_name}/{object_key}")
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size / 1024:.2f} KB")
        
        s3_client.upload_file(
            file_path, 
            bucket_name, 
            object_key,
            ExtraArgs={'ContentType': 'audio/midi'}
        )
        
        print("Upload successful!")
        
        # 生成预签名 URL
        download_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key
            },
            ExpiresIn=expire_seconds
        )
        
        expires_at = (datetime.utcnow() + timedelta(seconds=expire_seconds)).isoformat() + "Z"
        
        print(f"Generated presigned URL (expires in {expire_seconds}s)")
        
        return {
            "url": download_url,
            "key": object_key,
            "bucket": bucket_name,
            "expires_at": expires_at,
            "size_kb": round(file_size / 1024, 2)
        }
        
    except ClientError as e:
        print(f"S3 upload error: {e}")
        raise Exception(f"Failed to upload to S3: {str(e)}")
    except Exception as e:
        print(f"Error: {e}")
        raise

def handler(event):
    """
    RunPod Serverless Entry
    
    Expected input:
    {
        "input": {
            "audio_base64": "<base64 audio>",
            "expire_hours": 1  // S3 URL 过期时间（小时），默认 1
        }
    }
    
    Required Environment Variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION
    - S3_BUCKET_NAME
    """
    try:
        print("=" * 60)
        print("Received MT3 transcription request")
        
        # Parse input
        audio_b64 = event["input"]["audio_base64"]
        expire_hours = event["input"].get("expire_hours", 1)
        
        audio_bytes = base64.b64decode(audio_b64)
        audio_size_mb = len(audio_bytes) / 1024 / 1024
        
        print(f"Audio size: {audio_size_mb:.2f} MB")
        print(f"URL expiration: {expire_hours} hour(s)")
        print("=" * 60)

        # Get S3 bucket from environment
        bucket_name = os.environ.get("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable not set")

        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            audio_path = f.name
        
        print(f"Saved audio to {audio_path}")

        # Transcribe with MT3
        midi_path = transcribe_mt3(audio_path)
        print(f"Transcription complete")
        
        # Get MIDI file size
        midi_size = os.path.getsize(midi_path)
        print(f"MIDI file size: {midi_size / 1024:.2f} KB")
        
        # Generate unique object key
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        job_id = event.get("id", "unknown")[:8]
        object_key = f"MT3/{timestamp}_{job_id}.mid"
        
        # Upload to S3
        expire_seconds = expire_hours * 3600
        s3_info = upload_to_s3(midi_path, bucket_name, object_key, expire_seconds)
        
        # Cleanup
        try:
            os.unlink(audio_path)
            os.unlink(midi_path)
            print("Cleanup completed")
        except Exception as e:
            print(f"Cleanup warning: {e}")
        
        print("=" * 60)
        print("Request completed successfully!")
        print("=" * 60)
        
        return {
            "download_url": s3_info["url"],
            "s3_key": s3_info["key"],
            "s3_bucket": s3_info["bucket"],
            "expires_at": s3_info["expires_at"],
            "size_kb": s3_info["size_kb"],
            "model": "MT3",
            "instructions": "Download the MIDI file from download_url before it expires"
        }
    
    except Exception as e:
        print("=" * 60)
        print(f"ERROR: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting RunPod MT3 serverless handler with S3 storage...")
    print("Multi-track music transcription to MIDI")
    print("Storage: S3 with presigned URLs")
    runpod.serverless.start({"handler": handler})
