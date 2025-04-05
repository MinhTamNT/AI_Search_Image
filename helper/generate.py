from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image as PILImage

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_comments(image_path, num_comments=4):
    image = PILImage.open(image_path).convert('RGB')
    prompts = [
        "Describe this image briefly.",
        "Give a detailed description of the image.",
        "What is happening in this photo?",
        "Write a creative caption for this picture.",
        "Summarize the image in one sentence."
    ]

    comments = []
    for prompt in prompts[:num_comments]:
        inputs = caption_processor(images=image, text=prompt, return_tensors="pt")
        output = caption_model.generate(**inputs, max_length=50)
        caption = caption_processor.decode(output[0], skip_special_tokens=True)
        comments.append(caption)

    return comments

