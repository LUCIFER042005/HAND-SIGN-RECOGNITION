import io
import pickle
import hashlib
import os
import secrets
from datetime import datetime
import pymysql
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import mediapipe as mp
import numpy as np
from PIL import Image

app = FastAPI(title="Hand Sign Recognition API", version="1.0")

# Security handler for HTTP Basic Auth on Admin routes
security = HTTPBasic()

# --- ADMIN CREDENTIALS ---
ADMIN_USERNAME = "lucifer"
ADMIN_PASSWORD = "mysecretpassword123"  # Change this to your preferred admin password!


def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect admin username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


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


# --- Protected Admin Endpoint to View All Users (Styled Dashboard) ---
@app.get("/admin/users", response_class=HTMLResponse)
def get_all_users_dashboard(admin: str = Depends(authenticate_admin)):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, created_at FROM users")
        users = cursor.fetchall()
        conn.close()

        rows = ""
        for user in users:
            created_str = (
                user['created_at'].strftime("%b %d, %Y")
                if hasattr(user['created_at'], 'strftime')
                else str(user['created_at'])
            )
            rows += f"""
            <tr>
                <td>{user['id']}</td>
                <td class="user-name">{user['username']}</td>
                <td>{created_str}</td>
            </tr>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Admin Dashboard</title>
            <style>
                body {{ font-family: 'Courier New', Courier, monospace; background-color: #121212; color: #4af6c6; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }}
                .card {{ background: #1e1e1e; border: 2px solid #4af6c6; border-radius: 12px; padding: 25px; box-shadow: 0 10px 30px rgba(0,255,200,0.1); width: 100%; max-width: 550px; text-align: center; }}
                h1 {{ font-size: 20px; letter-spacing: 2px; margin-bottom: 10px; text-shadow: 0 0 8px rgba(74,246,198,0.4); }}
                .badge {{ font-size: 16px; color: #fff; background: #2a2a2a; padding: 8px 16px; border-radius: 8px; border: 1px solid #333; display: inline-block; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 12px; border: 1px solid #333; text-align: center; }}
                th {{ background-color: #2a2a2a; color: #4af6c6; font-size: 14px; text-transform: uppercase; }}
                td {{ color: #ddd; font-size: 14px; }}
                .user-name {{ color: #fff; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>╔══════════════════════╗<br>║   ADMIN DASHBOARD    ║<br>╚══════════════════════╝</h1>
                <div class="badge">👥 TOTAL USERS: {len(users)}</div>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>USERNAME</th>
                            <th>JOINED</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


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