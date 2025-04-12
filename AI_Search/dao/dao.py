import pandas as pd
import os
from sqlalchemy.orm import sessionmaker
from AI_Search.model import Image, ImageComment, Tag, engine
from AI_Search.helper.generate import generate_comments
from PIL import Image as PILImage
import imagehash
from keybert import KeyBERT

# Setup session
Session = sessionmaker(bind=engine)
session = Session()

# Khởi tạo mô hình KeyBERT
kw_model = KeyBERT()


def extract_tags_from_comments(comments, top_k=5):
    # Lọc bỏ NaN, chuyển hết sang chuỗi và loại bỏ khoảng trắng
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
    df = pd.read_csv(comments_file, low_memory=False)

    for img_path, embedding in zip(image_paths, embeddings):
        img_name = os.path.basename(img_path)
        img_hash = compute_image_hash(img_path)

        if is_duplicate_image(img_hash):
            print(f"Ảnh {img_name} đã tồn tại trong cơ sở dữ liệu. Bỏ qua ảnh này.")
            continue

        # Lưu ảnh
        img_embedding = Image(
            image_path=img_path,
            embedding=embedding.tolist(),
            image_hash=img_hash
        )
        session.add(img_embedding)
        session.commit()

        # Lấy comment tương ứng ảnh
        comments = df[df['image_name'] == img_name].sort_values(by='comment_number')

        if not comments.empty:
            # Làm sạch dữ liệu comment
            comment_texts = comments['comment'].fillna('').astype(str).str.strip().tolist()

            # Lưu comment
            comment_objs = [
                ImageComment(image_id=img_embedding.id, comment=text)
                for text in comment_texts
            ]
            session.bulk_save_objects(comment_objs)

            # Trích xuất tag từ comment
            tags = extract_tags_from_comments(comment_texts, top_k=5)

            # Kiểm tra tag đã tồn tại
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


def get_tags_with_pagination(page, per_page, search_name=None):

    query = session.query(Tag)

    if search_name:
        query = query.filter(Tag.tag_name.ilike(f"%{search_name}%"))

    total_results = query.count()
    tags = query.offset((page - 1) * per_page).limit(per_page).all()
    total_pages = (total_results + per_page - 1) // per_page

    return total_results, total_pages, tags
