import cloudinary
import cloudinary.uploader
import os

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def upload_image_to_cloudinary(local_path, folder="my_images"):
    """
    Upload ảnh lên Cloudinary vào folder chỉ định.
    :param local_path: Đường dẫn file ảnh local
    :param folder: Tên folder trên Cloudinary
    :return: url ảnh trên Cloudinary hoặc None nếu lỗi
    """
    try:
        result = cloudinary.uploader.upload(
            local_path,
            folder=folder,              
            use_filename=True,
            unique_filename=False,
            overwrite=True
        )
        return result.get("secure_url")
    except Exception as e:
        print(f"Lỗi upload lên Cloudinary: {e}")
        return None

