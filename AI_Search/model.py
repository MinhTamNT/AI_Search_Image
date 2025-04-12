from pgvector.sqlalchemy import VECTOR
from sqlalchemy import create_engine, Column, Integer, Text, ForeignKey ,String ,Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import os
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()

image_tag_association = Table(
    'Image_Tag_Association', Base.metadata,
    Column('image_id', Integer, ForeignKey('Image.id', ondelete="CASCADE")),
    Column('tag_id', Integer, ForeignKey('Tag.id', ondelete="CASCADE"))
)

class Image(Base):
    __tablename__ = 'Image'
    id = Column(Integer, primary_key=True)
    image_path = Column(Text, nullable=False)
    embedding = Column(VECTOR(2560), nullable=False)
    image_hash = Column(String, nullable=False, unique=True)
    comments = relationship("ImageComment", back_populates="image", cascade="all, delete-orphan")
    image_url = Column(Text, nullable=True)
    tags = relationship("Tag", secondary=image_tag_association, back_populates="images")

class Tag(Base):
    __tablename__ = 'Tag'
    id = Column(Integer, primary_key=True)
    tag_name = Column(String(100), nullable=False, unique=True)
    images = relationship("Image", secondary=image_tag_association, back_populates="tags")


class ImageComment(Base):
    __tablename__ = 'Image_Comment'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('Image.id', ondelete='CASCADE'), nullable=False)
    comment = Column(Text, nullable=False)
    image = relationship("Image", back_populates="comments")

engine = create_engine(os.getenv("DATABASE_URL"))

Base.metadata.create_all(engine)

