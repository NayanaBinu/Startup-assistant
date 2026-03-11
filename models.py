from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class User(db.Model):
    """
    Represents a user of the application.
    Each user can own multiple startups.
    """
    __tablename__ = "users"
    
    user_id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to the Startup model
    startups = db.relationship("Startup", back_populates="owner", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"


class Startup(db.Model):
    """
    Represents a startup profile linked to a user.
    Contains detailed information about the startup's business.
    """
    __tablename__ = "startups"
    
    startup_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    
    startup_name = db.Column(db.String(150), nullable=False)
    domain = db.Column(db.String(50), nullable=False)
    registration_type = db.Column(db.String(50), nullable=False)
    stage = db.Column(db.String(50), nullable=False)
    funding_amount = db.Column(db.Numeric(12, 2), comment="In Lakhs")
    team_size = db.Column(db.Integer)
    location = db.Column(db.String(100))
    website = db.Column(db.String(2048))
    problem_statement = db.Column(db.Text)
    vision = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to the User model
    owner = db.relationship("User", back_populates="startups")

    def __repr__(self):
        return f"<Startup {self.startup_name}>"