from datetime import timedelta

from fastapi import APIRouter, status, Body, HTTPException, Response, Depends

from app.models.user import UserModel, user_collection, UserResponseModel, UserBody
from app.utils.auth import create_access_token, get_current_user, verify_password

router = APIRouter()


@router.get(
    "/",
    response_description="Get logged in user",
    status_code=status.HTTP_200_OK,
    response_model_by_alias=True,
    response_model=UserResponseModel,
)
async def get_user(current_user=Depends(get_current_user)):
    user = user_collection.find_one({"email": current_user.email})

    return {"message": "You are logged in!", "user": user}


@router.post(
    "/register",
    response_description="Add new user",
    response_model=UserResponseModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=True,
)
async def create_user(response: Response, user: UserModel = Body(...)):
    """
    Insert a new user record.

    A unique `id` will be created and provided in the response.
    """
    try:
        existing_user = user_collection.find_one(
            {"$or": [{"email": user.email}, {"username": user.username}]}
        )
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email or username already exists",
            )

        new_user = user_collection.insert_one(
            user.model_dump(by_alias=True, exclude={"id"})
        )

        created_user = user_collection.find_one({"_id": new_user.inserted_id})

        access_token = create_access_token(
            {"sub": user.email}, timedelta(minutes=60 * 24)
        )

        # Set cookie
        response.set_cookie(
            "token", access_token, httponly=True, secure=False, samesite="lax"
        )

        return {
            "message": "User registered successfully",
            "user": created_user,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post(
    "/login",
    response_description="Login user",
    response_model=UserResponseModel,
    status_code=status.HTTP_200_OK,
    response_model_by_alias=True,
)
async def login(response: Response, user: UserBody = Body(...)):
    try:
        identifier = user.identifier
        db_user = user_collection.find_one(
            {"$or": [{"email": identifier}, {"username": identifier}]}
        )
        if not db_user or not verify_password(user.password, db_user["password"]):
            print(db_user["password"], user.password)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )

        access_token = create_access_token(
            {"sub": db_user["email"]}, timedelta(minutes=60 * 24)
        )

        response.set_cookie(
            "token", access_token, httponly=True, secure=False, samesite="lax"
        )

        return {
            "message": "User successfully logged in!",
            "user": db_user,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post(
    "/logout",
    response_description="Logout user",
    status_code=status.HTTP_200_OK,
)
async def logout(response: Response):
    response.delete_cookie("token")
    return {"message": "User successfully logged out!"}
