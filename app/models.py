from sqlalchemy import Column, Integer, String, LargeBinary, Text
from app.database import Base


class User(Base):
    __tablename__ = 'Usuarios'
    idUsuarios = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100))
    email = Column(String(150), unique=True)
    fotoPerfil = Column(String(255), nullable=True)  # URL de la imagen en Cloudinary

class FaceEmbedding(Base):
    __tablename__ = 'face_embeddings'
    id = Column(Integer, primary_key=True, index=True)
    image_url = Column(String(500), unique=True, index=True)
    embedding_json = Column(Text, nullable=False) # JSON string of the numpy array
