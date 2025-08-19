from PIL import Image
from nsfw_image_detector import NSFWDetector
print("NSFW内容" if NSFWDetector().is_nsfw(Image.open("porn.jpg")) else "安全内容")