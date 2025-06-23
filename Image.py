from PIL import Image as PILImage, ImageDraw, ImageFilter
from typing import Tuple, Optional, Union, Any
import numpy as np

class Image:
    """
    A comprehensive Image class wrapping PIL functionality for loading, saving, 
    basic manipulations, and conversion to/from numpy arrays.
    """
    
    def __init__(self, source: Union[str, PILImage.Image, np.ndarray]):
        """
        Initialize the Image object.
        - source: Path to image file, PIL Image object, or numpy ndarray.
        """
        if isinstance(source, str):
            self._img = PILImage.open(source)
        elif isinstance(source, PILImage.Image):
            self._img = source.copy()
        elif isinstance(source, np.ndarray):
            self._img = PILImage.fromarray(source)
        else:
            raise TypeError("Unsupported source type for Image")
    
    @classmethod
    def from_path(cls, path: str) -> "Image":
        """Load an image from a file path."""
        return cls(path)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Image":
        """Create an Image from a numpy array."""
        return cls(arr)
    
    def save(self, path: str, format: Optional[str] = None, **kwargs: Any) -> None:
        """
        Save image to a file.
        - path: Destination file path.
        - format: Image format (e.g., 'PNG', 'JPEG'). Inferred from path if None.
        - kwargs: Additional options passed to PIL save.
        """
        self._img.save(path, format=format, **kwargs)
    
    def show(self, title: Optional[str] = None) -> None:
        """Display the image using default image viewer."""
        self._img.show(title=title)
    
    def resize(self, size: Tuple[int, int], resample: int = PILImage.LANCZOS) -> "Image":
        """Return a resized copy of the image."""
        return Image(self._img.resize(size, resample=resample))
    
    def thumbnail(self, size: Tuple[int, int]) -> None:
        """Modify the image to contain a thumbnail version in-place."""
        self._img.thumbnail(size)
    
    def crop(self, box: Tuple[int, int, int, int]) -> "Image":
        """Return a cropped copy of the image. Box is (left, upper, right, lower)."""
        return Image(self._img.crop(box))
    
    def rotate(self, angle: float, expand: bool = True) -> "Image":
        """Return a rotated copy of the image."""
        return Image(self._img.rotate(angle, expand=expand))
    
    def convert(self, mode: str) -> "Image":
        """Return a copy converted to the given mode (e.g. 'RGB', 'L')."""
        return Image(self._img.convert(mode))
    
    def filter(self, filter_: ImageFilter.Filter) -> "Image":
        """Apply a PIL filter (e.g. BLUR, CONTOUR) and return the result."""
        return Image(self._img.filter(filter_))
    
    def draw(self) -> ImageDraw.ImageDraw:
        """Get a drawing context for the image."""
        return ImageDraw.Draw(self._img)
    
    def to_array(self) -> np.ndarray:
        """Convert the image to a numpy array."""
        return np.array(self._img)
    
    def get_pixel(self, x: int, y: int) -> Union[int, Tuple[int, ...]]:
        """Get the pixel value at (x, y)."""
        return self._img.getpixel((x, y))
    
    def set_pixel(self, x: int, y: int, value: Union[int, Tuple[int, ...]]) -> None:
        """Set the pixel at (x, y) to value."""
        self._img.putpixel((x, y), value)
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get image size as (width, height)."""
        return self._img.size
    
    @property
    def mode(self) -> str:
        """Get image mode (e.g. 'RGB', 'RGBA')."""
        return self._img.mode
    
    @property
    def format(self) -> Optional[str]:
        """Get image file format (e.g. 'JPEG')."""
        return self._img.format
    
    def blend(self, other: "Image", alpha: float) -> "Image":
        """
        Blend with another image of the same size.
        - other: Another Image object.
        - alpha: Blend factor (0.0: this only, 1.0: other only).
        """
        if self.size != other.size:
            raise ValueError("Images must be the same size to blend")
        blended = PILImage.blend(self._img, other._img, alpha)
        return Image(blended)
    
    def save_thumbnail(self, path: str, size: Tuple[int, int], **kwargs: Any) -> None:
        """
        Create and save a thumbnail of the image without modifying the original.
        - path: Destination file path.
        - size: Thumbnail size.
        """
        thumb = self._img.copy()
        thumb.thumbnail(size)
        thumb.save(path, **kwargs)
    
    def __repr__(self) -> str:
        return f"<Image size={self.size} mode={self.mode} format={self.format}>"
    
# Example usage:
if __name__ == "__main__":
    img = Image.from_path("example.jpg")
    print(img)
    resized = img.resize((200, 200))
    resized.save("resized.png")
    print("Saved resized image.")
