
from backend.database import Base, engine
from backend.models import Customer, Prediction

Base.metadata.create_all(bind=engine)

print("Database tables created successfully!")
