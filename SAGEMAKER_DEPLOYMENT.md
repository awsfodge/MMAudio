# MMAudio SageMaker Deployment Guide

This guide explains how to deploy MMAudio on Amazon SageMaker using the ModelBuilder tool.

## Overview

The deployment solution consists of:
- **`deploy_sagemaker.py`**: Main deployment script with custom InferenceSpec
- **`requirements_sagemaker.txt`**: Optimized dependencies for SageMaker (excludes heavy client-side packages)
- Support for local testing before cloud deployment
- **Pre-packaged model weights** for faster cold start (2-3 minutes vs 5-7 minutes)
- GPU-optimized inference

## Prerequisites

1. **AWS Account** with SageMaker access
2. **IAM Role** with SageMaker permissions (see setup instructions below)
3. **Python 3.9+** with dependencies installed:
   ```bash
   # Install MMAudio and its dependencies
   pip install -e .

   # Install SageMaker SDK for deployment (client-side only)
   pip install sagemaker>=2.200.0
   ```
   **Note**: The SageMaker SDK is only needed on your local machine for deployment. It's NOT included in the inference container (optimized for faster cold start).

4. **AWS CLI** configured with credentials:
   ```bash
   aws configure
   ```

### IAM Role Setup

Use the managed policy: `arn:aws:iam::aws:policy/AmazonSageMakerFullAccess`

OR

Your SageMaker execution role needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::sagemaker-*/*",
        "arn:aws:s3:::sagemaker-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
    }
  ]
}
```
## Quick Start

### 1. Local Testing (Recommended First Step)

Test the deployment locally before deploying to SageMaker:

```bash
python deploy_sagemaker.py --mode local --test
```

This will:
- Load the model locally
- Run a test inference with a sample text prompt
- Save the generated audio to `./output/test_sagemaker_output.flac`

### 2. Deploy to SageMaker

Once local testing succeeds, deploy to SageMaker:

```bash
python deploy_sagemaker.py \
    --mode sagemaker \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE \
    --instance-type ml.g5.xlarge \
    --instance-count 1
```

**What happens during deployment:**
1. Downloads model weights locally (~2-3 GB, takes 2-3 minutes)
2. Packages code + weights into a model artifact
3. Uploads to S3 (~3-6 GB depending on variant)
4. Creates SageMaker model, endpoint configuration, and endpoint
5. Launches inference container on ml.g5.xlarge instance

**Total deployment time**: ~10-15 minutes

The script will output the endpoint name when complete:
```
Deployment successful! Endpoint name: mmaudio-large-44k-v2-20251020-005229
```

**Important**: Save this endpoint name - you'll need it for inference!

### 3. Wait for Container Initialization

After deployment completes, wait **2-3 minutes** for the container to:
- Install Python dependencies
- Load model weights into GPU memory
- Initialize the inference server

You can monitor the endpoint status:
```bash
aws sagemaker describe-endpoint --endpoint-name <your-endpoint-name> --query 'EndpointStatus'
```

When it shows `"InService"`, wait an additional 2-3 minutes before your first request.

### 4. Test the Endpoint

Use the provided example script:

```bash
python3 example_sagemaker_inference.py \
    --endpoint-name <your-endpoint-name> \
    --mode text \
    --prompt "ocean waves crashing on the beach" \
    --output ocean_waves.flac
```

**First request**: May take 60-90 seconds (container is still warming up)
**Subsequent requests**: 30-60 seconds per 8-second audio generation

## Deployment Options

### Model Variants

Choose from different model sizes (larger = better quality, more GPU memory):

```bash
# Default: large_44k_v2 (best quality, ~6GB GPU memory)
python deploy_sagemaker.py --mode sagemaker --variant large_44k_v2 --role <role-arn>

# Medium model (balanced)
python deploy_sagemaker.py --mode sagemaker --variant medium_44k --role <role-arn>

# Small model (faster, less memory)
python deploy_sagemaker.py --mode sagemaker --variant small_44k --role <role-arn>
```

Available variants:
- `large_44k_v2` (default, recommended) - 44.1kHz output
- `large_44k` - 44.1kHz output
- `medium_44k` - 44.1kHz output
- `small_44k` - 44.1kHz output
- `small_16k` - 16kHz output (requires BigVGAN vocoder)

### Instance Types

The model requires GPU instances. Recommended options:

```bash
# Single GPU (recommended for most use cases)
--instance-type ml.g5.xlarge

# Larger GPU for better performance
--instance-type ml.g5.2xlarge

# Multi-GPU (requires code modifications for data parallelism)
--instance-type ml.g5.12xlarge
```

**Memory Requirements:**
- Minimum: ~6GB GPU memory (for large_44k_v2)
- Recommended: ml.g5.xlarge (24GB GPU memory)

### Custom Endpoint Name

You can specify a custom endpoint name (must be ≤ 63 characters):

```bash
python deploy_sagemaker.py \
    --mode sagemaker \
    --role <role-arn> \
    --endpoint-name mmaudio-prod
```

**Note**: If no name is provided, the script auto-generates one like: `mmaudio-large-44k-v2-20251020-005229`

## Using the Deployed Endpoint

### Command-Line Interface (Recommended)

The easiest way to test the endpoint is using the provided example script:

```bash
# Text-to-audio
python3 example_sagemaker_inference.py \
    --endpoint-name <your-endpoint-name> \
    --mode text \
    --prompt "ocean waves crashing on the beach" \
    --output ocean_waves.flac

# Video-to-audio
python3 example_sagemaker_inference.py \
    --endpoint-name <your-endpoint-name> \
    --mode video \
    --video /path/to/video.mp4 \
    --output video_audio.flac

# Image-to-audio
python3 example_sagemaker_inference.py \
    --endpoint-name <your-endpoint-name> \
    --mode image \
    --image /path/to/image.png \
    --prompt "birds chirping in a forest" \
    --output image_audio.flac
```

**All available parameters:**
```bash
python3 example_sagemaker_inference.py \
    --endpoint-name <name> \
    --mode {text,video,image} \
    --prompt "text description" \
    --video /path/to/video.mp4 \        # For video mode
    --image /path/to/image.png \        # For image mode
    --output output.flac \
    --negative-prompt "music" \         # What to avoid
    --duration 8.0 \                    # Audio duration (seconds)
    --cfg-strength 4.5 \                # Guidance 1-10 (higher = more adherence)
    --num-steps 25 \                    # Diffusion steps (higher = better quality)
    --seed 42                           # Random seed (-1 for random)
```

### Python SDK (Programmatic Access)

```python
import boto3
import json
import base64
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Connect to the endpoint
predictor = Predictor(
    endpoint_name='your-endpoint-name',
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

# Text-to-audio example
response = predictor.predict({
    'prompt': 'ocean waves crashing on a beach',
    'negative_prompt': 'music',
    'duration': 8.0,
    'cfg_strength': 4.5,
    'num_steps': 25,
    'seed': 42,
    'output_format': 'flac'
})

# Save the audio
audio_bytes = base64.b64decode(response['audio_base64'])
with open('output.flac', 'wb') as f:
    f.write(audio_bytes)

print(f"Generated audio: {response['sampling_rate']}Hz, {response['duration']}s")
```

### Video-to-audio example

```python
import base64

# Load and encode video
with open('input_video.mp4', 'rb') as f:
    video_base64 = base64.b64encode(f.read()).decode('utf-8')

# Generate audio for video
response = predictor.predict({
    'video_base64': video_base64,
    'prompt': 'footsteps on wooden floor',  # Optional, enhances the audio
    'negative_prompt': 'music',
    'duration': 8.0,
    'cfg_strength': 4.5,
    'num_steps': 25,
    'seed': 42
})

# Save the audio
audio_bytes = base64.b64decode(response['audio_base64'])
with open('video_audio.flac', 'wb') as f:
    f.write(audio_bytes)
```

### Image-to-audio example

```python
import base64

# Load and encode image
with open('input_image.png', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Generate audio for image
response = predictor.predict({
    'image_base64': image_base64,
    'prompt': 'birds chirping in a forest',
    'duration': 8.0,
    'cfg_strength': 4.5,
    'num_steps': 25
})

# Save the audio
audio_bytes = base64.b64decode(response['audio_base64'])
with open('image_audio.flac', 'wb') as f:
    f.write(audio_bytes)
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_base64` | string | - | Base64-encoded video file (MP4 recommended) |
| `image_base64` | string | - | Base64-encoded image file (PNG, JPG) |
| `prompt` | string | `""` | Text description of desired audio |
| `negative_prompt` | string | `""` | What to avoid in generation (e.g., "music") |
| `duration` | float | `8.0` | Audio duration in seconds |
| `cfg_strength` | float | `4.5` | Guidance strength (1-10, higher = more adherence to prompt) |
| `num_steps` | int | `25` | Number of diffusion steps (more = higher quality, slower) |
| `seed` | int | `-1` | Random seed (-1 for random) |
| `output_format` | string | `"flac"` | Audio format ("flac" or "wav") |

**Notes:**
- Either `video_base64`, `image_base64`, or `prompt` (or combinations) should be provided
- Video/image inputs are optional but enhance audio-visual synchronization
- Optimal duration is ~8 seconds (training duration)

## Output Format

The endpoint returns a JSON object:

```json
{
    "audio_base64": "base64-encoded-audio-data",
    "sampling_rate": 44100,
    "duration": 8.0,
    "format": "flac"
}
```

## Cost Estimation

Approximate SageMaker costs (us-east-1, as of 2024):

| Instance Type | GPU | Price/hour | Use Case |
|---------------|-----|------------|----------|
| ml.g5.xlarge | 1x A10G (24GB) | ~$1.41 | Production, recommended |
| ml.g5.2xlarge | 1x A10G (24GB) | ~$1.69 | Higher throughput |
| ml.g5.4xlarge | 1x A10G (24GB) | ~$2.25 | Maximum performance |

**Cold start time**: ~2-3 minutes (model weights are pre-packaged, only dependency installation needed)
**Inference time**: ~30-60 seconds for 8-second audio (depending on instance type and parameters)
**Note**: First invocation after deployment may take longer while the container initializes. Subsequent requests are faster.

## Troubleshooting

### Out of Memory (OOM)

- Use smaller model variant: `--variant small_44k`
- Use larger instance type: `--instance-type ml.g5.2xlarge`
- Reduce batch processing if modified

### Slow Inference

- Reduce `num_steps` (try 15-20 instead of 25)
- Ensure GPU instance is being used
- Use larger instance type

### Model Download Issues

- Model weights are now pre-packaged during deployment (no runtime download needed)
- If you see download errors during deployment, ensure your local machine has internet access and HuggingFace connectivity
- For custom VPC deployments, the endpoint no longer needs internet access for model downloads

### Endpoint Creation Fails

- Verify IAM role has necessary permissions:
  - `AmazonSageMakerFullAccess` (or similar)
  - S3 access for model artifacts
  - ECR access for container images
- Check instance type availability in your region
- Review CloudWatch logs for detailed error messages

### Endpoint Name Validation Error

If you see: `ValidationError: Member must have length less than or equal to 63`

**Cause**: Endpoint names are limited to 63 characters. The auto-generated names are now optimized to stay under this limit.

**Solutions**:
1. Use a custom short name: `--endpoint-name mmaudio-prod`
2. The deployment script now generates shorter names automatically
3. If you see this error, the endpoint may have been created with a truncated name - check:
   ```bash
   aws sagemaker list-endpoints --name-contains mmaudio
   ```

### Invocation Timeout

If your first request times out:
- **Wait 2-3 minutes** after endpoint shows "InService" before testing
- Container needs time to install dependencies and load models
- First request may take 60-90 seconds
- Check CloudWatch logs to see initialization progress:
  ```bash
  aws logs tail /aws/sagemaker/Endpoints/<endpoint-name> --follow
  ```

## Cleanup

**Important**: Delete endpoints when not in use to avoid ongoing charges (~$1.41/hour for ml.g5.xlarge)

### Option 1: AWS CLI (Recommended)

```bash
# Delete endpoint (this also cleans up endpoint config automatically)
aws sagemaker delete-endpoint --endpoint-name <your-endpoint-name>

# Optional: Delete the model (frees up S3 storage)
aws sagemaker delete-model --model-name <your-model-name>
```

### Option 2: Python SDK

```python
import boto3

sagemaker_client = boto3.client('sagemaker')

# Delete endpoint
sagemaker_client.delete_endpoint(EndpointName='your-endpoint-name')

# Delete endpoint configuration (if needed)
sagemaker_client.delete_endpoint_config(EndpointConfigName='your-endpoint-config-name')

# Delete model
sagemaker_client.delete_model(ModelName='your-model-name')
```

### Option 3: AWS Console

1. Navigate to **SageMaker → Endpoints**
2. Select your endpoint
3. **Actions → Delete**
4. Confirm deletion

**Note**: Deleting the endpoint stops billing immediately. Model artifacts remain in S3 (~3-6 GB) until manually deleted.

## Advanced Usage

### Cold Start Optimization

The deployment script now **automatically pre-downloads and packages model weights** during deployment. This optimization:

- **Reduces cold start time** from 5-7 minutes to 2-3 minutes
- **Eliminates runtime downloads** - weights are included in the model package uploaded to S3
- **Works in private VPCs** - endpoint doesn't need internet access for model downloads
- **Increases deployment time** by ~2-3 minutes (one-time cost during deployment)
- **Increases S3 storage** - model package is ~3-6GB depending on variant (instead of ~50MB)

The implementation automatically falls back to runtime downloads if pre-packaged weights are not found, ensuring backward compatibility.

### Custom Inference Parameters

Modify the inference parameters for different results:

- **Higher quality** (slower): `num_steps=50`, `cfg_strength=5.0`
- **Faster** (lower quality): `num_steps=15`, `cfg_strength=3.0`
- **More creative**: Lower `cfg_strength` (2.0-3.0)
- **More adherent to prompt**: Higher `cfg_strength` (5.0-7.0)

### Batch Processing

For processing multiple inputs, call the endpoint multiple times. The current implementation processes one input at a time.

### Monitoring

Monitor endpoint performance in CloudWatch:
- Model invocations
- Model latency
- CPU/GPU utilization
- Memory usage

## Example Workflow

Here's a complete end-to-end example:

```bash
# 1. Deploy the endpoint
python deploy_sagemaker.py \
    --mode sagemaker \
    --role arn:aws:iam::123456789012:role/SageMakerRole \
    --instance-type ml.g5.xlarge

# Output: Deployment successful! Endpoint name: mmaudio-large-44k-v2-20251020-123456

# 2. Wait 2-3 minutes for container initialization

# 3. Test with text-to-audio
python3 example_sagemaker_inference.py \
    --endpoint-name mmaudio-large-44k-v2-20251020-123456 \
    --mode text \
    --prompt "ocean waves crashing on the beach" \
    --output ocean_waves.flac

# 4. Generate more audio samples
python3 example_sagemaker_inference.py \
    --endpoint-name mmaudio-large-44k-v2-20251020-123456 \
    --mode text \
    --prompt "thunderstorm with heavy rain" \
    --output thunderstorm.flac \
    --duration 10.0 \
    --num-steps 50

# 5. When done, delete the endpoint to stop billing
aws sagemaker delete-endpoint --endpoint-name mmaudio-large-44k-v2-20251020-123456
```

## Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Deployment time | 10-15 min | One-time setup |
| Cold start time | 2-3 min | After endpoint shows "InService" |
| First inference | 60-90 sec | Container warming up |
| Subsequent inference | 30-60 sec | Per 8-second audio |
| Model package size | 3-6 GB | Uploaded to S3 |
| GPU memory required | ~6 GB | For large_44k_v2 |
| Cost per hour | ~$1.41 | ml.g5.xlarge in us-east-1 |

## Support

For issues or questions:
- **MMAudio**: https://github.com/hkchengrex/MMAudio
- **SageMaker SDK**: https://github.com/aws/sagemaker-python-sdk
- **This deployment**: Check CloudWatch logs for detailed error messages
- **AWS SageMaker Documentation**: https://docs.aws.amazon.com/sagemaker/

## License

This deployment script is provided as-is. The MMAudio model and its weights are subject to their respective licenses. See the main MMAudio repository for details.
