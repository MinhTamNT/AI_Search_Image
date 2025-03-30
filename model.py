from pgvector.sqlalchemy import VECTOR
from sqlalchemy import create_engine, Column, Integer, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv()
Base = declarative_base()



class Image(Base):
    __tablename__ = 'Image'
    id = Column(Integer, primary_key=True)
    image_path = Column(Text, nullable=False)
    embedding = Column(VECTOR(128))
    comments = relationship("ImageComment", back_populates="image", cascade="all, delete-orphan")

class ImageComment(Base):
    __tablename__ = 'Image_Comment'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('Image.id', ondelete='CASCADE'), nullable=False)
    comment = Column(Text, nullable=False)
    image = relationship("Image", back_populates="comments")

engine = create_engine(os.getenv("DATABASE_URL"))

Base.metadata.create_all(engine)