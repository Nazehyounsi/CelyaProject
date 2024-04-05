from transformers import pipeline
from PIL import Image
import requests

# Initialize the pipeline
pipe = pipeline("image-to-text", model="magistermilitum/tridis_HTR")

# Load an image
image_url = "Test.png"  # Replace this with your image URL
image = Image.open(requests.get(image_url, stream=True).raw)

# Run inference
result = pipe(image)

# Print the recognized text
print("Recognized text:", result[0]['generated_text'])

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import requests
#
# model_name = "magistermilitum/tridis_HTR"
# processor = TrOCRProcessor.from_pretrained(model_name)
# model = VisionEncoderDecoderModel.from_pretrained(model_name)
#
# image_url = "Test.png"  # Replace this with your image URL
# image = Image.open(requests.get(image_url, stream=True).raw)
#
# inputs = processor(images=image, return_tensors="pt")
# outputs = model.generate(**inputs)
# text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#
# print(text)