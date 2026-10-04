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
    
    def __init__(self, config_path, checkpoint_path, device='cpu',
                 onnx_path=None):
        """
        Initialize the single file inference pipeline.
        
        Args:
            config_path (str): Path to the model configuration file
            checkpoint_path (str): Path to the model checkpoint file
            device (str): Device to run inference on ('cpu' or 'cuda:0')
            onnx_path (str): if given, run inference through this ONNX model
                instead of torch. ~18x faster on the deployment Pi, whose
                torch build is 1.7.1 with 2020-era aarch64 kernels; outputs
                are numerically identical (100% argmax agreement).
        """
        if not check_dependencies():
            raise RuntimeError("Dependencies not available")
            
        self.device = device
        self.model = None
        self.sess = None
        self.classes = None
        self.test_pipeline = None
        self.palette = None

        # Initialize model
        if onnx_path:
            self._load_onnx(onnx_path, checkpoint_path)
        else:
            self._load_model(config_path, checkpoint_path)
        self._setup_pipeline()

    def _load_onnx(self, onnx_path, checkpoint_path):
        """Load an ONNX model exported by export_onnx_5band.py.

        The torch model is not built at all, so none of its weights are
        allocated. CLASSES and PALETTE still come from the checkpoint, which is
        read for its metadata only.
        """
        import onnxruntime as ort
        print(f"Loading ONNX model: {onnx_path}")
        self.sess = ort.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider'])
        meta = torch.load(checkpoint_path, map_location='cpu').get('meta', {})
        self.classes = meta.get('CLASSES') or ('background', 'water')
        self.palette = meta.get('PALETTE') or [[120, 120, 120], [204, 0, 102]]
        print(f"Model loaded successfully. Classes: {self.classes}")
        print(f"Palette: {self.palette}")
        print("Using device: cpu (onnxruntime)")
    
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
        
        self.classes = self.model.CLASSES
        print(f"Model loaded successfully. Classes: {self.classes}")
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

        if self.sess is not None:
            return self._inference_onnx(data)

        if next(self.model.parameters()).is_cuda:
            # Scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]
        
        # Run inference
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        
        return result[0]  # Return first (and only) result

    def _inference_onnx(self, data):
        """Run the ONNX model and finish the job mmseg would have done.

        The exported graph stops at the resized input's resolution, so the
        logits are cropped back from the pad, resized to the original shape
        and argmaxed here.
        """
        import torch.nn.functional as F

        img = data['img'][0]
        meta = data['img_metas'][0]
        meta = meta.data if hasattr(meta, 'data') else meta
        while isinstance(meta, (list, tuple)):
            meta = meta[0]

        # The export substitutes scale_factor for size in the decode head,
        # which is only exact when both dimensions divide by 32.
        h, w = int(img.shape[2]), int(img.shape[3])
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            # MEASURED CAVEAT: padding is not free. MiT's spatial-reduction
            # attention pools globally, so a padded strip shifts predictions
            # across the whole image, not just at the border.
            #
            # Measured over 64 real 5-band captures (Onondaga Lake Park, all
            # 1296x972 -> pipeline 512x683 -> padded 512x704), comparing torch
            # against torch with no ONNX involved:
            #
            #   pixels agreeing     mean 96.73%   range 91.37% - 98.79%
            #   water fraction      mean +0.89 pts, range -3.85 to +4.88
            #   shifted over 1 pt   39 of 64 images
            #
            # The direction is not systematic -- it moves both ways depending
            # on content. No padding mode avoids it; replicate and reflect
            # simply bias differently. The magnitude tracks how confident the
            # model is: the 40k-iteration type_pool model moves 0.79% of
            # pixels, this 100-iteration checkpoint 3.3% on average.
            #
            # Feed dimensions already divisible by 32 to avoid this entirely.
            print(f"  WARNING: input {h}x{w} is not divisible by 32; padding "
                  f"by ({ph},{pw}). This shifts predictions -- see the note in "
                  f"_inference_onnx. Prefer input sized to a multiple of 32.")
            img = F.pad(img, (0, pw, 0, ph))

        logits = torch.from_numpy(
            self.sess.run(None, {'input': img.numpy()})[0])
        if ph or pw:
            logits = logits[:, :, :h, :w]

        oh, ow = meta['ori_shape'][:2]
        logits = F.interpolate(logits, size=(oh, ow), mode='bilinear',
                               align_corners=False)
        return logits.argmax(1)[0].numpy().astype(np.uint8)
    
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
        
        # Save as PNG. This is the segmentation MASK, not a picture of one.
        #
        # It used to be written with plt.savefig(bbox_inches='tight', dpi=150),
        # which produced an RGBA rendering of the palette at whatever size
        # matplotlib chose — for a 972x1296 input, an 871x1162x4 image with 206
        # distinct values. SU-WaterCam/tools/segformer_daemon.py writes the same
        # logical output as a single-channel mask at source resolution, so the
        # two segmentation paths that are meant to be interchangeable were not:
        # tools/watercam.py only ever takes this one, and handed the render
        # straight to the LoRa bitmap compressor.
        #
        # Class indices are scaled across 0-255 exactly as the daemon does it,
        # so a mask from either path is the same file.
        if output_path.endswith('.png'):
            from PIL import Image as _Image

            mask = result.astype(np.uint8)
            n_classes = int(mask.max()) + 1 if self.palette is None else len(self.palette)
            if n_classes > 1:
                vis = np.round(mask.astype(np.float32) * (255.0 / (n_classes - 1))).astype(np.uint8)
            else:
                vis = mask
            _Image.fromarray(vis).save(output_path)

            # The colourised view is still useful for eyeballing a scene, so it
            # is kept — beside the mask, not instead of it.
            preview_path = output_path[:-len('.png')] + '_preview.png'
            plt.figure(figsize=(10, 8))
            plt.imshow(self._colorize_segmentation(result))
            plt.axis('off')
            plt.savefig(preview_path, bbox_inches='tight', pad_inches=0, dpi=150)
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
    parser.add_argument(
        '--onnx',
        type=str,
        default=None,
        help='Run inference through this ONNX model instead of torch '
             '(see export_onnx_5band.py); ~18x faster on CPU, same output'
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
            device=args.device,
            onnx_path=args.onnx
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
