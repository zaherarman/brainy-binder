from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import settings
from .models import Base

engine = None
session_factory = None

def init_db():
    """
    Init. the db engine and create tables

    Call during the app. start
    """

    global engine, session_factory
    db_url = f"sqlite:///{settings.sqlite_db_path}"

    # Object that knows where the db is and how to communicate with it
    engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
    
    Base.metadata.create_all(engine)
    
    session_factory = sessionmaker(bind=engine, expire_on_commit=False) # When called, returns a Session object with this engine

@contextmanager
def get_session():
    """
    Context manager for db sessions

    Yields:
        Database session
    """

    if session_factory is None:
        init_db()

    session = session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


    