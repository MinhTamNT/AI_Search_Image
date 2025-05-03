import pandas as pd
import os

from keras.src.layers.core import embedding
from sqlalchemy.orm import sessionmaker
from AI_Search.model import Image, ImageComment, Tag, engine
from AI_Search.helper.generate import generate_comments
from PIL import Image as PILImage
import imagehash
from keybert import KeyBERT
from AI_Search.Service.image_service import get_extract_model, get_image_embedding
from AI_Search.helper.upload_file import upload_image_to_cloudinary
Session = sessionmaker(bind=engine)
session = Session()

kw_model = KeyBERT()


def extract_tags_from_comments(comments, top_k=5):
    cleaned_comments = [str(c).strip() for c in comments if pd.notna(c)]
    combined_text = " ".join(cleaned_comments)

    if not combined_text:
        return []

    keywords = kw_model.extract_keywords(
        combined_text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_k
    )
    return [kw[0] for kw in keywords]


def compute_image_hash(img_path):
    pil_img = PILImage.open(img_path)
    img_hash = imagehash.phash(pil_img)
    return str(img_hash)


def is_duplicate_image(image_hash):
    existing_image = session.query(Image).filter_by(image_hash=image_hash).first()
    return existing_image is not None


def save_embeddings_and_comments_to_db(image_paths, embeddings, comments_file):
    df = pd.read_csv(comments_file, usecols=['image_name', 'comment_number', 'comment'], low_memory=False)

    for img_path, embedding in zip(image_paths, embeddings):
        img_name = os.path.basename(img_path)
        print(img_name)
        img_hash = compute_image_hash(img_path)
        print(img_name, img_hash)
        if is_duplicate_image(img_hash):
            print(f"Ảnh {img_name} đã tồn tại trong cơ sở dữ liệu. Bỏ qua ảnh này.")
            continue
        cloudinary_url = upload_image_to_cloudinary(img_path, folder="dataset_folder")
        img_embedding = Image(
            image_path=img_path,
            embedding=embedding.tolist(),
            image_hash=img_hash,
            image_url= cloudinary_url
        )
        session.add(img_embedding)
        session.commit()


        comments = df[df['image_name'] == img_name].sort_values(by='comment_number')

        if not comments.empty:
            # Làm sạch dữ liệu comment
            comment_texts = comments['comment'].fillna('').astype(str).str.strip().tolist()

            comment_objs = [
                ImageComment(image_id=img_embedding.id, comment=text)
                for text in comment_texts
            ]
            session.bulk_save_objects(comment_objs)

            tags = extract_tags_from_comments(comment_texts, top_k=5)

            existing_tags = {
                t.tag_name: t
                for t in session.query(Tag).filter(Tag.tag_name.in_(tags)).all()
            }

            tag_objs_to_add = []
            for tag in tags:
                if tag in existing_tags:
                    existing_tags[tag].images.append(img_embedding)
                else:
                    tag_obj = Tag(tag_name=tag, images=[img_embedding])
                    tag_objs_to_add.append(tag_obj)

            if tag_objs_to_add:
                session.add_all(tag_objs_to_add)

    session.commit()


def update_dataset(file_path, embedding):
    try:
        if not os.path.exists(file_path):
            print(f"File {file_path} không tồn tại.")
            return

        image_hash = compute_image_hash(file_path)
        if image_hash is None:
            print("Không thể tính toán hash cho ảnh.")
            return

        if is_duplicate_image(image_hash):
            print("Ảnh đã tồn tại trong cơ sở dữ liệu. Không lưu thêm.")
            return
        cloudinary_url = upload_image_to_cloudinary(file_path, folder="dataset_folder")
        new_image = Image(
            image_path=file_path,
            embedding=embedding.flatten(),
            image_hash=image_hash,
            image_url= cloudinary_url
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


def update_dataset(upload_folder_path):
    if not os.path.exists(upload_folder_path):
        print(f"Thư mục {upload_folder_path} không tồn tại")
        return
    
    image_paths = []
    
    for root, dirs, files in os.walk(upload_folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                img_path = os.path.join(root, file)
                image_paths.append(img_path)
    
    if not image_paths:
        print("Không tìm thấy ảnh nào trong thư mục upload")
        return
    
    extract_model = get_extract_model()
    
    for img_path in image_paths:
        try:
            image_hash = compute_image_hash(img_path)
            
            if is_duplicate_image(image_hash):
                print(f"Ảnh {img_path} đã tồn tại trong cơ sở dữ liệu")
                os.remove(img_path)
                continue
            
            img_embedding = get_image_embedding(img_path, extract_model)
            
            comments = generate_comments(img_path)
            
            tags = extract_tags_from_comments(comments)
            
            new_image = Image(
                image_path=img_path,
                embedding=img_embedding.flatten(),
                image_hash=image_hash
            )
            session.add(new_image)
            session.flush()
            
            for comment in comments:
                if pd.notna(comment):
                    new_comment = ImageComment(
                        image_id=new_image.id,
                        comment=comment
                    )
                    session.add(new_comment)
            
            for tag_text in tags:
                tag = session.query(Tag).filter_by(tag_name=tag_text).first()
                if not tag:
                    tag = Tag(tag_name=tag_text)
                    session.add(tag)
                    session.flush()
                
                new_image.tags.append(tag)
            
            print(f"Đã xử lý và thêm ảnh {img_path}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_path}: {str(e)}")
    
    session.commit()
    print(f"Đã cập nhật dataset với {len(image_paths)} ảnh từ thư mục upload")


def get_tags_with_pagination(page, per_page, search_name=None):

    query = session.query(Tag)

    if search_name:
        query = query.filter(Tag.tag_name.ilike(f"%{search_name}%"))

    total_results = query.count()
    tags = query.offset((page - 1) * per_page).limit(per_page).all()
    total_pages = (total_results + per_page - 1) // per_page

    return total_results, total_pages, tags
