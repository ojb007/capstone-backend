from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class Experiment(Base):
    __tablename__ = "experiments"

    id              = Column(Integer, primary_key=True, index=True)
    group_name      = Column(String, nullable=False)   # A~F
    dataset         = Column(String, nullable=False)   # fpb / fiqa / finqa
    model           = Column(String, nullable=False)
    prompt_strategy = Column(String, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

    results = relationship("Result", back_populates="experiment")


class Result(Base):
    __tablename__ = "results"

    id              = Column(Integer, primary_key=True, index=True)
    experiment_id   = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    accuracy        = Column(Float)
    f1_macro        = Column(Float)
    f1_micro        = Column(Float)
    avg_latency_ms  = Column(Float)
    total_cost_usd  = Column(Float)
    cost_per_item   = Column(Float)

    experiment = relationship("Experiment", back_populates="results")