import pandas as pd
import os
from sqlalchemy.orm import sessionmaker
from model import Image, ImageComment, engine
from helper.generate import generate_comments

Session = sessionmaker(bind=engine)
session = Session()


def save_embeddings_and_comments_to_db(image_paths, embeddings, comments_file):
    df = pd.read_csv(comments_file ,low_memory=False)

    for img_path, embedding in zip(image_paths, embeddings):
        img_name = os.path.basename(img_path)
        img_embedding = Image(image_path=img_path, embedding=embedding.tolist())
        session.add(img_embedding)
        session.commit()

        comments = df[df['image_name'] == img_name].sort_values(by='comment_number')

        comments_to_add = [
            ImageComment(image_id=img_embedding.id, comment=row['comment'])
            for _, row in comments.iterrows()
        ]

        session.bulk_save_objects(comments_to_add)
    session.commit()


def update_dataset(file_path, embedding):
    new_image = Image(image_path=file_path, embedding=embedding.flatten())
    session.add(new_image)
    session.commit()

    try:
        comment_texts = generate_comments(file_path, num_comments=4)
        for cmt in comment_texts:
            comment = ImageComment(image_id=new_image.id, comment=cmt)
            session.add(comment)
        session.commit()
    except Exception as e:
        print(f"Error generating comments: {e}")

