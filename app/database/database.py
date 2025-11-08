from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./plant_doctor.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    
    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String)
    disease_name = Column(String)
    confidence = Column(Float)
    treatment = Column(String)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Créer les tables
Base.metadata.create_all(bind=engine)