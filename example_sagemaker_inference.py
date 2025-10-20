"""
Example script demonstrating how to use the deployed MMAudio SageMaker endpoint.

This script shows how to:
1. Connect to the deployed endpoint
2. Generate audio from text prompts
3. Generate audio from videos
4. Generate audio from images
"""

import argparse
import base64
import json
from pathlib import Path

import boto3
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


def text_to_audio(predictor: Predictor, prompt: str, output_path: str, **kwargs):
    """
    Generate audio from a text prompt.

    Args:
        predictor: SageMaker predictor instance
        prompt: Text description of the desired audio
        output_path: Path to save the generated audio
        **kwargs: Additional parameters (negative_prompt, duration, cfg_strength, etc.)
    """
    print(f"Generating audio from text prompt: '{prompt}'")

    # Prepare input
    input_data = {
        'prompt': prompt,
        'negative_prompt': kwargs.get('negative_prompt', 'music'),
        'duration': kwargs.get('duration', 8.0),
        'cfg_strength': kwargs.get('cfg_strength', 4.5),
        'num_steps': kwargs.get('num_steps', 25),
        'seed': kwargs.get('seed', -1),
        'output_format': kwargs.get('output_format', 'flac')
    }

    # Call endpoint
    response = predictor.predict(input_data)

    # Save audio
    audio_bytes = base64.b64decode(response['audio_base64'])
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)

    print(f"Audio saved to: {output_path}")
    print(f"Sampling rate: {response['sampling_rate']}Hz, Duration: {response['duration']}s")

    return response


def video_to_audio(predictor: Predictor, video_path: str, output_path: str, prompt: str = "", **kwargs):
    """
    Generate audio from a video file.

    Args:
        predictor: SageMaker predictor instance
        video_path: Path to input video file
        output_path: Path to save the generated audio
        prompt: Optional text prompt to enhance the audio
        **kwargs: Additional parameters
    """
    print(f"Generating audio from video: {video_path}")

    # Load and encode video
    with open(video_path, 'rb') as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Prepare input
    input_data = {
        'video_base64': video_base64,
        'prompt': prompt,
        'negative_prompt': kwargs.get('negative_prompt', 'music'),
        'duration': kwargs.get('duration', 8.0),
        'cfg_strength': kwargs.get('cfg_strength', 4.5),
        'num_steps': kwargs.get('num_steps', 25),
        'seed': kwargs.get('seed', -1),
        'output_format': kwargs.get('output_format', 'flac')
    }

    # Call endpoint
    print("Calling SageMaker endpoint (this may take 30-60 seconds)...")
    response = predictor.predict(input_data)

    # Save audio
    audio_bytes = base64.b64decode(response['audio_base64'])
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)

    print(f"Audio saved to: {output_path}")
    print(f"Sampling rate: {response['sampling_rate']}Hz, Duration: {response['duration']}s")

    return response


def image_to_audio(predictor: Predictor, image_path: str, output_path: str, prompt: str = "", **kwargs):
    """
    Generate audio from an image file.

    Args:
        predictor: SageMaker predictor instance
        image_path: Path to input image file
        output_path: Path to save the generated audio
        prompt: Text prompt describing the desired audio
        **kwargs: Additional parameters
    """
    print(f"Generating audio from image: {image_path}")

    # Load and encode image
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Prepare input
    input_data = {
        'image_base64': image_base64,
        'prompt': prompt,
        'negative_prompt': kwargs.get('negative_prompt', 'music'),
        'duration': kwargs.get('duration', 8.0),
        'cfg_strength': kwargs.get('cfg_strength', 4.5),
        'num_steps': kwargs.get('num_steps', 25),
        'seed': kwargs.get('seed', -1),
        'output_format': kwargs.get('output_format', 'flac')
    }

    # Call endpoint
    print("Calling SageMaker endpoint (this may take 30-60 seconds)...")
    response = predictor.predict(input_data)

    # Save audio
    audio_bytes = base64.b64decode(response['audio_base64'])
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)

    print(f"Audio saved to: {output_path}")
    print(f"Sampling rate: {response['sampling_rate']}Hz, Duration: {response['duration']}s")

    return response


def main():
    """Main function with example usage."""
    parser = argparse.ArgumentParser(
        description='Example usage of MMAudio SageMaker endpoint'
    )

    parser.add_argument(
        '--endpoint-name',
        type=str,
        required=True,
        help='Name of the deployed SageMaker endpoint'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['text', 'video', 'image'],
        default='text',
        help='Generation mode'
    )

    parser.add_argument(
        '--prompt',
        type=str,
        default='ocean waves crashing on the beach',
        help='Text prompt for audio generation'
    )

    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video file (required for video mode)'
    )

    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image file (required for image mode)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output.flac',
        help='Output audio file path'
    )

    parser.add_argument(
        '--negative-prompt',
        type=str,
        default='music',
        help='Negative prompt (what to avoid)'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=8.0,
        help='Audio duration in seconds'
    )

    parser.add_argument(
        '--cfg-strength',
        type=float,
        default=4.5,
        help='Classifier-free guidance strength (1-10)'
    )

    parser.add_argument(
        '--num-steps',
        type=int,
        default=25,
        help='Number of diffusion steps'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='Random seed (-1 for random)'
    )

    args = parser.parse_args()

    # Connect to endpoint
    print(f"Connecting to SageMaker endpoint: {args.endpoint_name}")
    predictor = Predictor(
        endpoint_name=args.endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    # Common parameters
    params = {
        'negative_prompt': args.negative_prompt,
        'duration': args.duration,
        'cfg_strength': args.cfg_strength,
        'num_steps': args.num_steps,
        'seed': args.seed,
        'output_format': 'flac'
    }

    # Generate audio based on mode
    if args.mode == 'text':
        text_to_audio(predictor, args.prompt, args.output, **params)

    elif args.mode == 'video':
        if not args.video:
            parser.error("--video is required for video mode")
        video_to_audio(predictor, args.video, args.output, args.prompt, **params)

    elif args.mode == 'image':
        if not args.image:
            parser.error("--image is required for image mode")
        image_to_audio(predictor, args.image, args.output, args.prompt, **params)

    print("\nDone!")


if __name__ == '__main__':
    main()
