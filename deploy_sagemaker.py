"""
MMAudio SageMaker Deployment Script using ModelBuilder

This script deploys the MMAudio model to Amazon SageMaker using the ModelBuilder tool.
It supports video-to-audio, image-to-audio, and text-to-audio synthesis.

Usage:
    # Local testing
    python deploy_sagemaker.py --mode local --test

    # Deploy to SageMaker
    python deploy_sagemaker.py --mode sagemaker --role <execution-role-arn>
"""

import argparse
import base64
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sagemaker import Session
from sagemaker.serve import ModelBuilder, SchemaBuilder, InferenceSpec
from sagemaker.serve.mode.function_pointers import Mode

# Note: torch, torchaudio, and MMAudio imports are done inside methods to avoid pickling issues

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class MMAudioInferenceSpec(InferenceSpec):
    """
    Custom InferenceSpec for MMAudio model deployment on SageMaker.

    This class handles loading all MMAudio components and performing inference
    for video-to-audio, image-to-audio, and text-to-audio synthesis.
    """

    def __init__(self, model_variant: str = 'large_44k_v2'):
        """
        Initialize the inference spec with the specified model variant.

        Args:
            model_variant: Model variant to use (small_16k, small_44k, medium_44k,
                          large_44k, large_44k_v2)
        """
        super().__init__()
        # Only store picklable configuration (string)
        self.model_variant = model_variant

    def load(self, model_dir: str):
        """
        Load the MMAudio model and all required components.

        This method is called by SageMaker when the endpoint is created.
        It downloads and initializes all model components.

        Args:
            model_dir: Directory where model artifacts are stored

        Returns:
            Dictionary containing all loaded models
        """
        # Import torch and MMAudio modules locally to avoid pickling issues
        import torch
        from mmaudio.eval_utils import ModelConfig, all_model_cfg
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils

        log.info(f"Loading MMAudio model variant: {self.model_variant}")

        # Determine device
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
            log.info("Using CUDA")
        elif torch.backends.mps.is_available():
            device = 'mps'
            log.info("Using MPS")
        else:
            log.warning('CUDA/MPS not available, using CPU')

        # Use bfloat16 for inference
        dtype = torch.bfloat16

        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Get model configuration
        if self.model_variant not in all_model_cfg:
            raise ValueError(f'Unknown model variant: {self.model_variant}')

        model_config: ModelConfig = all_model_cfg[self.model_variant]

        # Check if weights were pre-packaged (optimization for faster cold start)
        weights_dir = Path(model_dir) / 'weights'
        if weights_dir.exists():
            log.info(f"Using pre-packaged model weights from {weights_dir}")
            # Override paths to use pre-packaged weights
            model_config.model_path = str(weights_dir / Path(model_config.model_path).name)
            model_config.vae_path = str(weights_dir / Path(model_config.vae_path).name)
            model_config.synchformer_ckpt = str(weights_dir / Path(model_config.synchformer_ckpt).name)
            if model_config.bigvgan_16k_path:
                model_config.bigvgan_16k_path = str(weights_dir / Path(model_config.bigvgan_16k_path).name)
        else:
            # Fallback to downloading weights at runtime (slower cold start)
            log.info("Pre-packaged weights not found. Downloading model weights from HuggingFace...")
            model_config.download_if_needed()

        seq_cfg = model_config.seq_cfg

        # Load main MMAudio network
        log.info("Loading MMAudio network...")
        net: MMAudio = get_my_mmaudio(model_config.model_name).to(
            device, dtype
        ).eval()
        net.load_weights(
            torch.load(model_config.model_path, map_location=device, weights_only=True)
        )
        log.info(f'Loaded weights from {model_config.model_path}')

        # Load feature extraction utilities (CLIP, Synchformer, VAE)
        log.info("Loading feature extraction models (CLIP, Synchformer, VAE)...")
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=model_config.vae_path,
            synchformer_ckpt=model_config.synchformer_ckpt,
            enable_conditions=True,
            mode=model_config.mode,
            bigvgan_vocoder_ckpt=model_config.bigvgan_16k_path,
            need_vae_encoder=False
        )
        feature_utils = feature_utils.to(device, dtype).eval()

        log.info("All models loaded successfully!")

        return {
            'net': net,
            'feature_utils': feature_utils,
            'seq_cfg': seq_cfg,
            'model_config': model_config,
            'device': device,
            'dtype': dtype
        }

    def invoke(self, input_data: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform inference on the input data.

        Args:
            input_data: Dictionary containing:
                - video_base64 (optional): Base64-encoded video file
                - image_base64 (optional): Base64-encoded image file
                - prompt (optional): Text prompt for audio generation
                - negative_prompt (optional): Negative text prompt
                - duration (optional): Audio duration in seconds (default: 8.0)
                - cfg_strength (optional): Classifier-free guidance strength (default: 4.5)
                - num_steps (optional): Number of diffusion steps (default: 25)
                - seed (optional): Random seed for reproducibility (default: -1 for random)
                - output_format (optional): Output audio format, 'flac' or 'wav' (default: 'flac')
            model: Dictionary of loaded models from load() method

        Returns:
            Dictionary containing:
                - audio_base64: Base64-encoded audio file
                - sampling_rate: Audio sampling rate
                - duration: Actual audio duration
        """
        # Import torch, torchaudio, and MMAudio modules locally to avoid pickling issues
        import torch
        import torchaudio
        from mmaudio.eval_utils import generate, load_image, load_video
        from mmaudio.model.flow_matching import FlowMatching

        log.info("Starting inference...")

        # Extract models from the model dict
        net = model['net']
        feature_utils = model['feature_utils']
        seq_cfg = model['seq_cfg']
        device = model['device']

        # Extract parameters from input
        prompt = input_data.get('prompt', '')
        negative_prompt = input_data.get('negative_prompt', '')
        duration = input_data.get('duration', 8.0)
        cfg_strength = input_data.get('cfg_strength', 4.5)
        num_steps = input_data.get('num_steps', 25)
        seed = input_data.get('seed', -1)
        output_format = input_data.get('output_format', 'flac')

        # Setup random seed
        rng = torch.Generator(device=device)
        if seed >= 0:
            rng.manual_seed(seed)
            log.info(f"Using seed: {seed}")
        else:
            rng.seed()

        # Setup flow matching
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

        # Process input video/image
        clip_frames = None
        sync_frames = None

        if 'video_base64' in input_data:
            log.info("Processing video input...")
            # Decode base64 video
            video_bytes = base64.b64decode(input_data['video_base64'])

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
                tmp_video.write(video_bytes)
                tmp_video_path = tmp_video.name

            try:
                # Load video
                video_info = load_video(Path(tmp_video_path), duration)
                clip_frames = video_info.clip_frames.unsqueeze(0)
                sync_frames = video_info.sync_frames.unsqueeze(0)
                duration = video_info.duration_sec
            finally:
                # Clean up temporary file
                os.unlink(tmp_video_path)

        elif 'image_base64' in input_data:
            log.info("Processing image input...")
            # Decode base64 image
            image_bytes = base64.b64decode(input_data['image_base64'])

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_image:
                tmp_image.write(image_bytes)
                tmp_image_path = tmp_image.name

            try:
                # Load image
                image_info = load_image(Path(tmp_image_path))
                clip_frames = image_info.clip_frames.unsqueeze(0)
                sync_frames = image_info.sync_frames.unsqueeze(0)
                # For images, we use the specified duration directly
            finally:
                # Clean up temporary file
                os.unlink(tmp_image_path)

        # Update sequence configuration
        seq_cfg.duration = duration
        net.update_seq_lengths(
            seq_cfg.latent_seq_len,
            seq_cfg.clip_seq_len,
            seq_cfg.sync_seq_len
        )

        log.info(f"Generating audio with prompt: '{prompt}', duration: {duration}s")

        # Generate audio
        with torch.inference_mode():
            audios = generate(
                clip_frames,
                sync_frames,
                [prompt] if prompt else None,
                negative_text=[negative_prompt] if negative_prompt else None,
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength
            )

        # Convert to CPU and extract first audio
        audio = audios.float().cpu()[0]

        # Save audio to temporary file (torchaudio with torchcodec backend requires a file path)
        with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp_file:
            tmp_audio_path = tmp_file.name

        try:
            torchaudio.save(tmp_audio_path, audio, seq_cfg.sampling_rate)

            # Read back into memory
            with open(tmp_audio_path, 'rb') as f:
                audio_bytes = f.read()
        finally:
            # Clean up temporary file
            os.unlink(tmp_audio_path)

        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        log.info(f"Inference completed successfully. Audio duration: {duration}s")

        return {
            'audio_base64': audio_base64,
            'sampling_rate': seq_cfg.sampling_rate,
            'duration': duration,
            'format': output_format
        }


def create_sample_input():
    """Create a sample input for schema builder and testing."""
    return {
        'prompt': 'ocean waves crashing on the beach',
        'negative_prompt': 'music',
        'duration': 8.0,
        'cfg_strength': 4.5,
        'num_steps': 25,
        'seed': 42,
        'output_format': 'flac'
    }


def create_sample_output():
    """Create a sample output for schema builder."""
    # Create a minimal sample output without using torch/torchaudio to avoid pickling issues
    # Just use a fake base64 string - the schema builder only needs the structure, not real data
    fake_audio_base64 = "dGVzdF9hdWRpb19kYXRh"  # base64 encoded "test_audio_data"

    return {
        'audio_base64': fake_audio_base64,
        'sampling_rate': 44100,
        'duration': 8.0,
        'format': 'flac'
    }


def deploy_local(model_variant: str = 'large_44k_v2', test: bool = True):
    """
    Deploy the model locally for testing.

    Args:
        model_variant: Model variant to use
        test: Whether to run a test inference
    """
    log.info("Deploying MMAudio locally...")

    # Create inference spec
    inference_spec = MMAudioInferenceSpec(model_variant=model_variant)

    # Create schema builder
    schema_builder = SchemaBuilder(
        sample_input=create_sample_input(),
        sample_output=create_sample_output()
    )

    # Create model builder
    model_builder = ModelBuilder(
        mode=Mode.LOCAL_CONTAINER,
        inference_spec=inference_spec,
        schema_builder=schema_builder,
        model_path=str(Path.cwd()),  # Use current directory as model path
    )

    # Build the model
    log.info("Building model in local mode...")
    model = model_builder.build()

    if test:
        log.info("Running test inference...")

        # Create test input
        test_input = create_sample_input()

        # Run inference
        result = model.predict(test_input)

        log.info(f"Test inference completed! Sampling rate: {result['sampling_rate']}, "
                f"Duration: {result['duration']}s")

        # Optionally save the audio to a file
        audio_bytes = base64.b64decode(result['audio_base64'])
        output_path = Path('./output/test_sagemaker_output.flac')
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        log.info(f"Test audio saved to: {output_path}")

    return model


def deploy_sagemaker(
    role_arn: str,
    model_variant: str = 'large_44k_v2',
    instance_type: str = 'ml.g5.xlarge',
    instance_count: int = 1,
    endpoint_name: Optional[str] = None
):
    """
    Deploy the model to Amazon SageMaker.

    Args:
        role_arn: AWS IAM role ARN for SageMaker
        model_variant: Model variant to use
        instance_type: EC2 instance type for the endpoint (must be GPU)
        instance_count: Number of instances for the endpoint
        endpoint_name: Optional custom endpoint name

    Returns:
        SageMaker predictor object
    """
    import sagemaker
    from datetime import datetime

    log.info("Deploying MMAudio to SageMaker...")

    # Generate a short endpoint name if not provided (must be <= 63 chars)
    if endpoint_name is None:
        # Format: mmaudio-{variant}-{timestamp}
        # Example: mmaudio-large-20251020-005229 (34 chars)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        variant_short = model_variant.replace('_', '-')[:20]  # Limit variant name
        endpoint_name = f"mmaudio-{variant_short}-{timestamp}"

        # Ensure we're under the 63 character limit
        if len(endpoint_name) > 63:
            endpoint_name = endpoint_name[:63]

        log.info(f"Generated endpoint name: {endpoint_name} ({len(endpoint_name)} chars)")

    # Create inference spec
    inference_spec = MMAudioInferenceSpec(model_variant=model_variant)

    # Create schema builder
    schema_builder = SchemaBuilder(
        sample_input=create_sample_input(),
        sample_output=create_sample_output()
    )

    # Create SageMaker session
    sagemaker_session = Session()

    # Get the PyTorch inference container image URI
    # Using PyTorch 2.5 with Python 3.11 (adjust version as needed)
    image_uri = sagemaker.image_uris.retrieve(
        framework='pytorch',
        region=sagemaker_session.boto_region_name,
        version='2.5.1',
        py_version='py311',
        instance_type=instance_type,
        image_scope='inference'
    )
    log.info(f"Using container image: {image_uri}")

    # Create model directory and pre-download weights for faster cold start
    import shutil
    model_dir = Path(tempfile.mkdtemp(prefix='mmaudio_deploy_'))

    try:
        log.info(f"Creating model package in {model_dir}")

        # Copy only the source code directory (mmaudio/)
        src_dir = Path.cwd() / 'mmaudio'
        if src_dir.exists():
            shutil.copytree(src_dir, model_dir / 'mmaudio',
                          ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))

        # Copy requirements file
        requirements_file = Path.cwd() / 'requirements_sagemaker.txt'
        if requirements_file.exists():
            shutil.copy(requirements_file, model_dir / 'requirements_sagemaker.txt')

        # Download model weights locally before packaging (major cold start optimization!)
        log.info(f"Downloading model weights for {model_variant} to include in package...")
        log.info("This will take a few minutes but dramatically reduces endpoint cold start time.")

        # Import here to avoid issues
        from mmaudio.eval_utils import all_model_cfg

        if model_variant not in all_model_cfg:
            raise ValueError(f'Unknown model variant: {model_variant}')

        model_config = all_model_cfg[model_variant]

        # Download all required weights
        model_config.download_if_needed()

        # Create weights directory in the package
        weights_dir = model_dir / 'weights'
        weights_dir.mkdir(exist_ok=True)

        # Copy downloaded weights to the package
        weight_files = [
            (model_config.model_path, 'Main model'),
            (model_config.vae_path, 'VAE'),
            (model_config.synchformer_ckpt, 'Synchformer'),
        ]

        if model_config.bigvgan_16k_path:
            weight_files.append((model_config.bigvgan_16k_path, 'BigVGAN vocoder'))

        for weight_path, description in weight_files:
            if weight_path and Path(weight_path).exists():
                dest_path = weights_dir / Path(weight_path).name
                log.info(f"  Copying {description}: {Path(weight_path).name} ({Path(weight_path).stat().st_size / (1024*1024):.1f} MB)")
                shutil.copy(weight_path, dest_path)
            else:
                log.warning(f"  Warning: {description} not found at {weight_path}")

        log.info(f"Model package size: {sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024*1024):.2f} MB")

        # Create model builder with the minimal directory
        model_builder = ModelBuilder(
            mode=Mode.SAGEMAKER_ENDPOINT,
            inference_spec=inference_spec,
            schema_builder=schema_builder,
            role_arn=role_arn,
            sagemaker_session=sagemaker_session,
            instance_type=instance_type,
            image_uri=image_uri,  # Explicitly specify the container image
            model_path=str(model_dir),  # Use minimal model directory
            dependencies={'requirements': 'requirements_sagemaker.txt'}
        )

        # Build the model
        log.info("Building model for SageMaker deployment...")
        model = model_builder.build()

        # Deploy to SageMaker endpoint
        log.info(f"Deploying to SageMaker endpoint with instance type: {instance_type}")
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name
        )

        log.info(f"Deployment successful! Endpoint name: {predictor.endpoint_name}")

        return predictor

    finally:
        # Clean up temporary model directory
        if model_dir.exists():
            shutil.rmtree(model_dir)
            log.info(f"Cleaned up temporary model directory: {model_dir}")


def main():
    """Main function to parse arguments and deploy the model."""
    parser = argparse.ArgumentParser(description='Deploy MMAudio to Amazon SageMaker')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'sagemaker'],
        default='local',
        help='Deployment mode: local testing or SageMaker deployment'
    )

    parser.add_argument(
        '--variant',
        type=str,
        default='large_44k_v2',
        choices=['small_16k', 'small_44k', 'medium_44k', 'large_44k', 'large_44k_v2'],
        help='MMAudio model variant to deploy'
    )

    parser.add_argument(
        '--role',
        type=str,
        help='AWS IAM role ARN for SageMaker (required for sagemaker mode)'
    )

    parser.add_argument(
        '--instance-type',
        type=str,
        default='ml.g5.xlarge',
        help='SageMaker instance type (must be GPU)'
    )

    parser.add_argument(
        '--instance-count',
        type=int,
        default=1,
        help='Number of instances for the endpoint'
    )

    parser.add_argument(
        '--endpoint-name',
        type=str,
        help='Custom endpoint name (optional)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test inference after deployment (local mode only)'
    )

    args = parser.parse_args()

    if args.mode == 'local':
        deploy_local(model_variant=args.variant, test=args.test)
    elif args.mode == 'sagemaker':
        if not args.role:
            parser.error("--role is required for sagemaker mode")

        deploy_sagemaker(
            role_arn=args.role,
            model_variant=args.variant,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            endpoint_name=args.endpoint_name
        )


if __name__ == '__main__':
    main()
