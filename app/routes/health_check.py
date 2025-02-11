from fastapi import APIRouter, status

router = APIRouter()

@router.get(
    "/",
    response_description="Health check",
    status_code=status.HTTP_200_OK,
)
def health_check():
    """
    Health check endpoint to verify the application's status.

    This route returns a 200 OK response with a simple JSON payload
    indicating the application's health status. It serves as a basic
    check to ensure the app is up and running without any errors.
    """
    return {"status": "ok"}