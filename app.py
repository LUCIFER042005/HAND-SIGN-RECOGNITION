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

security = HTTPBasic()

# --- ADMIN CREDENTIALS ---
ADMIN_USERNAME = "lucifer"
ADMIN_PASSWORD = "mysecretpassword123"


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


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
MODEL_PATH = os.path.join(BASE_DIR, "model.p")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB", "defaultdb")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 15747)) if os.getenv("MYSQL_PORT") else 15747


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
        ssl={'ssl': {}}
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
                created_at DATETIME NOT NULL,
                last_login DATETIME NULL
            )
        """)
        # Ensure column exists if table was created previously without last_login
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN last_login DATETIME NULL")
        except Exception:
            pass  # Column already exists

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                review TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)
        conn.close()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Database initialization error: {e}")


class UserAuth(BaseModel):
    username: str
    password: str


class ChangePassword(BaseModel):
    username: str
    old_password: str
    new_password: str


class UserReview(BaseModel):
    username: str
    review: str


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


model = None
hands = None


@app.on_event("startup")
def load_resources():
    global model, hands
    init_db()

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_dict = pickle.load(f)
            model = model_dict["model"]
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
    )


@app.post("/register")
def register(user: UserAuth):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed_pwd = hash_password(user.password)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            "INSERT INTO users (username, password, created_at, last_login) VALUES (%s, %s, %s, %s)",
            (user.username, hashed_pwd, created_at, created_at)
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
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed_pwd = hash_password(user.password)

        cursor.execute(
            "SELECT id, username FROM users WHERE username = %s AND password = %s",
            (user.username, hashed_pwd)
        )
        db_user = cursor.fetchone()

        if db_user:
            # Record login timestamp
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("UPDATE users SET last_login = %s WHERE id = %s", (now_str, db_user["id"]))
            conn.close()
            return {"success": True, "username": db_user["username"]}
        else:
            conn.close()
            raise HTTPException(status_code=401, detail="Invalid username or password.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/users/change-password")
def change_password(data: ChangePassword):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed_old = hash_password(data.old_password)
        hashed_new = hash_password(data.new_password)

        cursor.execute(
            "UPDATE users SET password = %s WHERE username = %s AND password = %s",
            (hashed_new, data.username, hashed_old)
        )
        affected = cursor.rowcount
        conn.close()

        if affected > 0:
            return {"success": True, "message": "Password updated successfully!"}
        else:
            raise HTTPException(status_code=400, detail="Incorrect current password.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/users/submit-review")
def submit_review(data: UserReview):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            "INSERT INTO reviews (username, review, created_at) VALUES (%s, %s, %s)",
            (data.username, data.review, created_at)
        )
        conn.close()
        return {"success": True, "message": "Review submitted successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/users/delete-me")
def delete_own_account(user: UserAuth):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed_pwd = hash_password(user.password)

        cursor.execute(
            "DELETE FROM users WHERE username = %s AND password = %s",
            (user.username, hashed_pwd)
        )
        affected_rows = cursor.rowcount
        conn.close()

        if affected_rows > 0:
            return {"success": True, "message": "Account deleted successfully."}
        else:
            raise HTTPException(status_code=401, detail="Invalid password verification.")
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


@app.delete("/admin/users/{user_id}")
def delete_user_by_admin(user_id: int, admin: str = Depends(authenticate_admin)):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        affected_rows = cursor.rowcount
        conn.close()

        if affected_rows > 0:
            return {"success": True, "message": f"User {user_id} deleted."}
        else:
            raise HTTPException(status_code=404, detail="User not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/admin/users", response_class=HTMLResponse)
def get_all_users_dashboard(admin: str = Depends(authenticate_admin)):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, created_at, last_login FROM users ORDER BY id ASC")
        users = cursor.fetchall()

        cursor.execute("SELECT username, review, created_at FROM reviews ORDER BY id DESC LIMIT 10")
        reviews = cursor.fetchall()
        conn.close()

        # Build table rows with Serial Number and Last Login Timestamp
        rows = ""
        for index, user in enumerate(users, start=1):
            created_str = (
                user['created_at'].strftime("%b %d, %Y")
                if hasattr(user['created_at'], 'strftime')
                else str(user['created_at'])
            )

            if user.get('last_login'):
                last_login_str = (
                    user['last_login'].strftime("%b %d, %Y %I:%M %p")
                    if hasattr(user['last_login'], 'strftime')
                    else str(user['last_login'])
                )
            else:
                last_login_str = "Never"

            rows += f"""
            <tr>
                <td><b>{index}</b></td>
                <td class="user-name">{user['username']}</td>
                <td>{created_str}</td>
                <td style="color:#4af6c6;">{last_login_str}</td>
                <td>
                    <button class="btn-del" onclick="deleteUser({user['id']}, '{user['username']}')">Delete</button>
                </td>
            </tr>
            """

        review_rows = ""
        for r in reviews:
            rev_date = r['created_at'].strftime("%b %d, %I:%M %p") if hasattr(r['created_at'], 'strftime') else str(
                r['created_at'])
            review_rows += f"""
            <div style="background:#2a2a2a; border:1px solid #333; border-radius:6px; padding:10px; margin-bottom:8px; text-align:left;">
                <div style="font-size:12px; color:#4af6c6; display:flex; justify-content:space-between;">
                    <b>@{r['username']}</b> <span>{rev_date}</span>
                </div>
                <div style="font-size:13px; color:#ddd; margin-top:4px;">"{r['review']}"</div>
            </div>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Admin Dashboard</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #4af6c6; display: flex; justify-content: center; align-items: flex-start; padding: 40px 0; min-height: 100vh; margin: 0; }}
                .card {{ background: #1e1e1e; border: 2px solid #4af6c6; border-radius: 12px; padding: 25px; box-shadow: 0 10px 30px rgba(0,255,200,0.1); width: 100%; max-width: 780px; text-align: center; }}
                h1 {{ font-size: 20px; letter-spacing: 2px; margin-bottom: 10px; text-shadow: 0 0 8px rgba(74,246,198,0.4); }}
                .badge {{ font-size: 16px; color: #fff; background: #2a2a2a; padding: 8px 16px; border-radius: 8px; border: 1px solid #333; display: inline-block; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 25px; }}
                th, td {{ padding: 12px; border: 1px solid #333; text-align: center; }}
                th {{ background-color: #2a2a2a; color: #4af6c6; font-size: 13px; text-transform: uppercase; }}
                td {{ color: #ddd; font-size: 13px; }}
                .user-name {{ color: #fff; font-weight: bold; }}
                .btn-del {{ background: #ff4d4d; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-family: inherit; font-weight: bold; }}
                .btn-del:hover {{ background: #e03e3e; }}
            </style>
            <script>
                async function deleteUser(id, username) {{
                    if (confirm("Delete account for user '" + username + "'?")) {{
                        const res = await fetch('/admin/users/' + id, {{ method: 'DELETE' }});
                        if (res.ok) {{
                            alert("User deleted!");
                            window.location.reload();
                        }} else {{
                            alert("Failed to delete user.");
                        }}
                    }}
                }}
            </script>
        </head>
        <body>
            <div class="card">
                <h1>╔══════════════════════╗<br>║   ADMIN DASHBOARD    ║<br>╚══════════════════════╝</h1>
                <div class="badge">👥 TOTAL USERS: {len(users)}</div>
                <table>
                    <thead>
                        <tr>
                            <th>SR NO.</th>
                            <th>USERNAME</th>
                            <th>JOINED</th>
                            <th>LAST LOGIN</th>
                            <th>ACTION</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>

                <h3 style="color:#fff; border-top: 1px solid #333; padding-top:20px; margin-top:20px; text-align:left;">💬 USER REVIEWS</h3>
                {review_rows if review_rows else '<div style="color:#888; font-size:13px;">No reviews submitted yet.</div>'}
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/")
def serve_frontend():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return {"error": "index.html file missing on server"}


@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": model is not None}


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