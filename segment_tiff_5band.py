#!/home/pi/miniforge3/envs/5band/bin/python3 
# use 5band env created by miniforge3 using SegformerDeps repo
"""
Single file inference script for 5-band flood segmentation.

This script processes a single 5-band TIFF file and saves the result in the same directory.

batch_inference_5band.py /path/to/file.tiff
"""

import argparse
import os
import sys
import glob
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add current directory to path to import mmseg modules
sys.path.append(os.getcwd())

# Try to import required modules
try:
    import mmcv
    import torch
    import rasterio
    from mmseg.apis import init_segmentor
    from mmseg.datasets.pipelines import Compose
    from mmseg.datasets.pipelines.loading import Load_5band_ImageFromFile
    from mmseg.datasets.pipelines.transforms import Normalize_5band
    from mmcv.parallel import collate, scatter
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies are not available: {e}")
    print("Please install required packages:")
    print("pip install mmcv torch rasterio matplotlib")
    DEPENDENCIES_AVAILABLE = False


def check_dependencies():
    """Check if all required dependencies are available."""
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Required dependencies are not installed.")
        print("Please install them using:")
        print("pip install mmcv torch rasterio matplotlib")
        return False
    return True


class SingleFileInference5Band:
    """Single file inference class for 5-band flood segmentation."""
    
    def __init__(self, config_path, checkpoint_path, device='cpu'):
        """
        Initialize the single file inference pipeline.
        
        Args:
            config_path (str): Path to the model configuration file
            checkpoint_path (str): Path to the model checkpoint file
            device (str): Device to run inference on ('cpu' or 'cuda:0')
        """
        if not check_dependencies():
            raise RuntimeError("Dependencies not available")
            
        self.device = device
        self.model = None
        self.test_pipeline = None
        self.palette = None
        
        # Initialize model
        self._load_model(config_path, checkpoint_path)
        self._setup_pipeline()
    
    def _load_model(self, config_path, checkpoint_path):
        """Load the segmentation model."""
        print(f"Loading model from config: {config_path}")
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Initialize model
        self.model = init_segmentor(config_path, checkpoint_path, device=self.device)
        self.model.eval()
        
        # Get palette for visualization
        if hasattr(self.model, 'PALETTE') and self.model.PALETTE is not None:
            self.palette = self.model.PALETTE
        else:
            # Default palette for binary segmentation (background, water)
            self.palette = [[120, 120, 120], [204, 0, 102]]
        
        print(f"Model loaded successfully. Classes: {self.model.CLASSES}")
        print(f"Palette: {self.palette}")
        print(f"Using device: {self.device}")
    
    def _setup_pipeline(self):
        """Setup the data preprocessing pipeline."""
        # Define the test pipeline for 5-band images
        self.test_pipeline = [
            dict(type='Load_5band_ImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize_5band',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]
        self.test_pipeline = Compose(self.test_pipeline)
    
    def _preprocess_image(self, image_path):
        """
        Preprocess a single 5-band image.
        
        Args:
            image_path (str): Path to the 5-band TIFF file
            
        Returns:
            dict: Preprocessed data ready for model inference
        """
        # Load image using rasterio
        with rasterio.open(image_path) as src:
            img = src.read()  # Shape: (bands, height, width)
            img = img.transpose(1, 2, 0)  # Shape: (height, width, bands)
            
            # Get metadata
            transform = src.transform
            crs = src.crs
            
        # Prepare data dictionary
        data = {
            'img': img,
            'img_info': {'filename': image_path},
            'ori_filename': os.path.basename(image_path)
        }
        
        # Apply preprocessing pipeline
        data = self.test_pipeline(data)
        
        return data, transform, crs
    
    def _inference_single(self, data):
        """
        Run inference on a single preprocessed image.
        
        Args:
            data (dict): Preprocessed image data
            
        Returns:
            numpy.ndarray: Segmentation result
        """
        # Prepare data for model
        data = collate([data], samples_per_gpu=1)
        
        if next(self.model.parameters()).is_cuda:
            # Scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]
        
        # Run inference
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        
        return result[0]  # Return first (and only) result
    
    def _save_result(self, result, output_path, transform=None, crs=None):
        """
        Save segmentation result to file.
        
        Args:
            result (numpy.ndarray): Segmentation result
            output_path (str): Output file path
            transform: Geotransform information
            crs: Coordinate reference system
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there is a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        # Save as PNG for visualization
        if output_path.endswith('.png'):
            # Create colored segmentation map
            colored_result = self._colorize_segmentation(result)
            # Use non-interactive matplotlib backend
            plt.figure(figsize=(10, 8))
            plt.imshow(colored_result)
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()  # Close figure to free memory
        
        # Save as GeoTIFF if transform and CRS are provided
        elif output_path.endswith('.tif'):
            if transform is not None and crs is not None:
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=result.shape[0],
                    width=result.shape[1],
                    count=1,
                    dtype=result.dtype,
                    crs=crs,
                    transform=transform
                ) as dst:
                    dst.write(result, 1)
            else:
                # Save as regular TIFF without geospatial info
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=result.shape[0],
                    width=result.shape[1],
                    count=1,
                    dtype=result.dtype
                ) as dst:
                    dst.write(result, 1)
    
    def _colorize_segmentation(self, seg_map):
        """
        Colorize segmentation map using the model's palette.
        
        Args:
            seg_map (numpy.ndarray): Segmentation map
            
        Returns:
            numpy.ndarray: Colored segmentation map
        """
        if self.palette is None:
            return seg_map
        
        # Create color map
        colors = np.array(self.palette, dtype=np.uint8)
        colored_map = colors[seg_map]
        
        return colored_map
    
    def process_single_file(self, input_path):
        """
        Process a single TIFF file and save result in the same directory.
        
        Args:
            input_path (str): Path to input TIFF file
        """
        print(f"Processing file: {input_path}")
        
        # Generate output path in the same directory
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_segmentation.png"
        
        # Preprocess image
        data, transform, crs = self._preprocess_image(input_path)
        
        # Run inference
        result = self._inference_single(data)
        
        # Save result
        self._save_result(result, str(output_path), transform, crs)
        
        print(f"Processed: {input_path} -> {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Single file inference for 5-band flood segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single TIFF file
    python batch_inference_5band.py /path/to/file.tiff
    
    # Use GPU if available
    python batch_inference_5band.py /path/to/file.tiff --device cuda:0
        """
    )
    
    # Input file (first positional argument)
    parser.add_argument(
        'input_file',
        type=str,
        help='Input 5-band TIFF file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default='./local_configs/segformer/B0/segformer.b0.512x512.flood_5band_2cls.65k.test.py',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./work_dirs/segformer.b0.512x512.flood_5band_2cls.65k_huantao/iter_100.pth',
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to run inference on (cpu, cuda:0, cuda:1) (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Validate that input file is not empty
    if not args.input_file.strip():
        parser.error("Input file cannot be empty")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        parser.error(f"Input file not found: {args.input_file}")
    
    # Check if input file is a TIFF file
    if not (args.input_file.lower().endswith('.tif') or args.input_file.lower().endswith('.tiff')):
        parser.error(f"Input file must be a TIFF file (.tif or .tiff): {args.input_file}")
    
    # Check if files exist
    if args.config and not os.path.exists(args.config):
        parser.error(f"Config file not found: {args.config}")
    if args.checkpoint and not os.path.exists(args.checkpoint):
        parser.error(f"Checkpoint file not found: {args.checkpoint}")
    
    # Initialize single file inference
    try:
        inference = SingleFileInference5Band(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        return 1
    
    # Process file
    try:
        inference.process_single_file(args.input_file)
        
        print("Processing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == '__main__':
    exit(main()) 
