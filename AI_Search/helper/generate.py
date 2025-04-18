from transformers import BlipProcessor, TFBlipForConditionalGeneration
from PIL import Image as PILImage

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

prompts = [
    "a beautiful scenery of",
    "an artistic representation of",
    "a detailed photograph of",
    "a stunning image of"
]

def generate_comments(image_path, num_comments=4):
    image = PILImage.open(image_path).convert('RGB')
    comments = []
    for prompt in prompts[:num_comments]:
        inputs = caption_processor(images=image, text=prompt, return_tensors="tf")
        output = caption_model.generate(**inputs, max_length=50)
        caption = caption_processor.decode(output[0], skip_special_tokens=True)
        comments.append(caption)

    return comments