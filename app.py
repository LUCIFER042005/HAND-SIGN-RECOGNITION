import io
import pickle
import hashlib
import os
from datetime import datetime
import pymysql
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import mediapipe as mp
import numpy as np
from PIL import Image

app = FastAPI(title="Hand Sign Recognition API", version="1.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve absolute path for project files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
MODEL_PATH = os.path.join(BASE_DIR, "model.p")

# --- MySQL Configuration from Environment Variables ---
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB", "defaultdb")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306)) if os.getenv("MYSQL_PORT") else 3306


def get_db_connection():
    if not MYSQL_HOST:
        return None
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        port=MYSQL_PORT,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        ssl={'ssl': {}}  # Enables SSL required by cloud hosts like Aiven
    )


def init_db():
    if not MYSQL_HOST:
        print("MYSQL_HOST environment variable not found. Database initialization skipped.")
        return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)
        conn.close()
        print("Permanent Cloud MySQL Database initialized successfully!")
    except Exception as e:
        print(f"Database initialization error: {e}")


class UserAuth(BaseModel):
    username: str
    password: str


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# --- Global ML Model Variables ---
model = None
hands = None


@app.on_event("startup")
def load_resources():
    global model, hands
    init_db()

    # Load trained model using absolute path
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_dict = pickle.load(f)
            model = model_dict["model"]
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
    )
    print("Model and MediaPipe initialized successfully!")


# --- Authentication Endpoints ---
@app.post("/register")
def register(user: UserAuth):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server (MYSQL_HOST).")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed_pwd = hash_password(user.password)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            "INSERT INTO users (username, password, created_at) VALUES (%s, %s, %s)",
            (user.username, hashed_pwd, created_at)
        )
        conn.close()
        return {"success": True, "message": "Account created successfully!"}
    except pymysql.err.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/login")
def login(user: UserAuth):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server (MYSQL_HOST).")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed_pwd = hash_password(user.password)

        cursor.execute(
            "SELECT id, username FROM users WHERE username = %s AND password = %s",
            (user.username, hashed_pwd)
        )
        db_user = cursor.fetchone()
        conn.close()

        if db_user:
            return {"success": True, "username": db_user["username"]}
        else:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/users/count")
def get_user_count():
    if not MYSQL_HOST:
        return {"total_users": 0}
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM users")
        result = cursor.fetchone()
        conn.close()
        return {"total_users": result["total"]}
    except Exception as e:
        return {"total_users": 0, "error": str(e)}


# --- Serve Frontend Page ---
@app.get("/")
def serve_frontend():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return {"error": "index.html file missing on server"}


@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": model is not None}


# --- Prediction Endpoint ---
@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    results = hands.process(frame)

    if not results.multi_hand_landmarks:
        return {
            "success": False,
            "prediction": None,
            "message": "No hand landmarks detected",
        }

    data_aux = []
    x_ = []
    y_ = []

    hand_landmarks = results.multi_hand_landmarks[0]

    for i in range(len(hand_landmarks.landmark)):
        x_.append(hand_landmarks.landmark[i].x)
        y_.append(hand_landmarks.landmark[i].y)

    for i in range(len(hand_landmarks.landmark)):
        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
        data_aux.append(hand_landmarks.landmark[i].y - min(y_))

    prediction = model.predict([np.asarray(data_aux)])
    predicted_character = str(prediction[0])

    return {
        "success": True,
        "prediction": predicted_character,
        "bbox": {
            "x_min": min(x_),
            "y_min": min(y_),
            "x_max": max(x_),
            "y_max": max(y_),
        },
    }