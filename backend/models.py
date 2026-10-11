
from sqlalchemy import Column, Integer, Float, String, DateTime
from backend.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, unique=True, index=True, nullable=False)
    recency = Column(Integer, nullable=False)
    frequency = Column(Integer, nullable=False)
    monetary = Column(Float, nullable=False)
    total_quantity = Column(Integer, nullable=False)
    unique_products = Column(Integer, nullable=False)
    repeat_purchase = Column(Integer, nullable=True)
    cluster = Column(Integer, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, nullable=True)
    predicted_repeat_purchase = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False)
