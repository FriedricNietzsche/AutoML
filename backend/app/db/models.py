from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from .session import Base

class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    stages = relationship("Stage", back_populates="project")

class Stage(Base):
    __tablename__ = 'stages'

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    stage_id = Column(String, index=True)
    status = Column(String, index=True)
    project = relationship("Project", back_populates="stages")

class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    content = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False)  # Unix timestamp
    project = relationship("Project")