from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    """Baser class for all database models"""
    pass

# Defining orm model
class Document(Base):
    """
    Document metadata model.    

    Stored metadata about documents such as path, type, title, timestamps and tags.
    """

    __tablename__ = "Documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, index=True)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)