import pandas as pd
import os
from sqlalchemy.orm import sessionmaker
from AI_Search.model import Image, ImageComment, engine
from AI_Search.helper.generate import generate_comments
from PIL import Image as PILImage
import imagehash
Session = sessionmaker(bind=engine)
session = Session()


def compute_image_hash(img_path):
    pil_img = PILImage.open(img_path)
    img_hash = imagehash.phash(pil_img)
    return str(img_hash)


def is_duplicate_image(image_hash):
    existing_image = session.query(Image).filter_by(image_hash=image_hash).first()
    return existing_image is not None


def save_embeddings_and_comments_to_db(image_paths, embeddings, comments_file):
    df = pd.read_csv(comments_file, low_memory=False)

    for img_path, embedding in zip(image_paths, embeddings):
        img_name = os.path.basename(img_path)

        img_hash = compute_image_hash(img_path)

        if is_duplicate_image(img_hash):
            print(f"Ảnh {img_name} đã tồn tại trong cơ sở dữ liệu. Bỏ qua ảnh này.")
            continue

        img_embedding = Image(image_path=img_path, embedding=embedding.tolist(), image_hash=img_hash)
        session.add(img_embedding)
        session.commit()

        comments = df[df['image_name'] == img_name].sort_values(by='comment_number')

        if not comments.empty:
            comments_to_add = [
                ImageComment(image_id=img_embedding.id, comment=row['comment'])
                for _, row in comments.iterrows()
            ]
            session.bulk_save_objects(comments_to_add)

    session.commit()


def update_dataset(file_path, embedding):
    try:
        image_hash = compute_image_hash(file_path)
        if image_hash is None:
            print("Không thể tính toán hash cho ảnh.")
            return
        if not os.path.exists(file_path):
            print(f"File {file_path} không tồn tại.")
            return
        if is_duplicate_image(image_hash):
            print("Ảnh đã tồn tại trong cơ sở dữ liệu. Không lưu thêm.")
            return

        new_image = Image(
            image_path=file_path,
            embedding=embedding.flatten(),
            image_hash=image_hash
        )
        session.add(new_image)
        session.commit()
        print("Ảnh mới đã được lưu vào database.")

        comment_texts = generate_comments(file_path, num_comments=4)
        for cmt in comment_texts:
            comment = ImageComment(image_id=new_image.id, comment=cmt)
            session.add(comment)
        session.commit()
        print("Đã lưu comment cho ảnh mới.")

    except Exception as e:
        print(f"Lỗi khi update dataset: {e}")


