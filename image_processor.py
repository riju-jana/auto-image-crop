"""
Image processing utilities for Smart Auto Crop AI
Handles image validation, conversion, and basic operations
"""
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Optional


class ImageProcessor:
    """Handles image processing operations"""
    
    MAX_SIZE_MB = 10
    ALLOWED_FORMATS = ['jpg', 'jpeg', 'png']
    
    @staticmethod
    def validate_image(file_content: bytes, filename: str) -> Tuple[bool, str]:
        """
        Validate uploaded image
        
        Args:
            file_content: Image file bytes
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > ImageProcessor.MAX_SIZE_MB:
            return False, f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({ImageProcessor.MAX_SIZE_MB}MB)"
        
        # Check file extension
        ext = filename.lower().split('.')[-1]
        if ext not in ImageProcessor.ALLOWED_FORMATS:
            return False, f"File format '{ext}' not allowed. Allowed formats: {', '.join(ImageProcessor.ALLOWED_FORMATS)}"
        
        # Try to load image
        try:
            img = Image.open(io.BytesIO(file_content))
            img.verify()
            return True, ""
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    @staticmethod
    def bytes_to_cv2(file_content: bytes) -> np.ndarray:
        """
        Convert bytes to OpenCV image
        
        Args:
            file_content: Image file bytes
            
        Returns:
            OpenCV image (numpy array)
        """
        nparr = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    @staticmethod
    def cv2_to_base64(img: np.ndarray, format: str = 'PNG') -> str:
        """
        Convert OpenCV image to base64 string
        
        Args:
            img: OpenCV image
            format: Output format (PNG or JPEG)
            
        Returns:
            Base64 encoded string
        """
        _, buffer = cv2.imencode(f'.{format.lower()}', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_base64}"
    
    @staticmethod
    def crop_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Crop image to specified coordinates
        
        Args:
            img: OpenCV image
            x, y: Top-left corner coordinates
            w, h: Width and height
            
        Returns:
            Cropped image
        """
        return img[y:y+h, x:x+w]
    
    @staticmethod
    def resize_for_processing(img: np.ndarray, max_dimension: int = 1024) -> Tuple[np.ndarray, float]:
        """
        Resize image for faster processing while maintaining aspect ratio
        
        Args:
            img: OpenCV image
            max_dimension: Maximum dimension (width or height)
            
        Returns:
            Tuple of (resized_image, scale_factor)
        """
        h, w = img.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= max_dimension:
            return img, 1.0
        
        scale = max_dimension / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    @staticmethod
    def draw_crop_box(img: np.ndarray, x: int, y: int, w: int, h: int, 
                      color: Tuple[int, int, int] = (0, 255, 0), 
                      thickness: int = 3) -> np.ndarray:
        """
        Draw crop box on image
        
        Args:
            img: OpenCV image
            x, y: Top-left corner
            w, h: Width and height
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn box
        """
        img_copy = img.copy()
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        return img_copy
