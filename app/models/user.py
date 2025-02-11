from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator, root_validator, model_validator
from pydantic.functional_validators import BeforeValidator
from datetime import datetime
from app.config import db

from typing_extensions import Annotated

from app.utils.auth import pwd_context, get_password_hash

PyObjectId = Annotated[str, BeforeValidator(str)]

# User Schema

class UserModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    name: str = Field(...)
    email: EmailStr = Field(...)
    username: str = Field(..., min_length=3)
    password: str = Field(...)
    credits: int = Field(default=5, ge=0)
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True

    @model_validator(mode="before")
    def hash_password(cls, values):
        """
        Hash the password before saving the user.
        """
        password = values.get("password")
        if password:
            values["password"] = get_password_hash(password)  # Hash the password
        return values

class UserBody(BaseModel):
    identifier: EmailStr | str = Field(...)
    password: str = Field(...)

    class Config:
        from_attributes = True


class UserPublicModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    name: str
    email: EmailStr
    username: str
    credits: int
    created_at: datetime

    class Config:
        from_attributes = True


class UserResponseModel(BaseModel):
    message: Optional[str] = None
    user: UserPublicModel
    access_token: str
    token_type: str

    class Config:
        from_attributes = True


user_collection = db.get_collection("users")
