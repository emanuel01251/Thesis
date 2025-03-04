from vercel_python import asgi
from app import app  # Import your Gradio FastAPI instance

handler = asgi.ASGIHandler(app)