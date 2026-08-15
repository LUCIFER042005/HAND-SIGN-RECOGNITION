import io
import pickle
import hashlib
import os
import secrets
from datetime import datetime, timezone, timedelta
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
ADMIN_USERNAME = "Punjan"
ADMIN_PASSWORD = "Punjan123"

# --- TIMEZONE CONFIGURATION (IST) ---
IST = timezone(timedelta(hours=5, minutes=30))


def convert_to_ist_str(dt_val):
    """Format stored datetime string/object directly without double-shifting timezone."""
    if not dt_val:
        return "-"
    if isinstance(dt_val, str):
        try:
            dt_val = datetime.strptime(dt_val, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return dt_val
    return dt_val.strftime("%b %d, %Y, %I:%M %p IST")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


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
TUTORIAL_PATH = os.path.join(BASE_DIR, "tutorial.html")
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
                created_at DATETIME NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                review TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_analytics (
                predicted_char VARCHAR(20) PRIMARY KEY,
                count INT DEFAULT 1
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
        created_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            "INSERT INTO users (username, password, created_at) VALUES (%s, %s, %s)",
            (user.username, user.password, created_at)
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
            """
            SELECT id, username FROM users 
            WHERE username = %s AND (password = %s OR password = %s)
            """,
            (user.username, user.password, hashed_pwd)
        )
        db_user = cursor.fetchone()
        conn.close()

        if db_user:
            return {"success": True, "username": db_user["username"]}
        else:
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

        cursor.execute(
            """
            UPDATE users SET password = %s 
            WHERE username = %s AND (password = %s OR password = %s)
            """,
            (data.new_password, data.username, data.old_password, hashed_old)
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
        created_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

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
            """
            DELETE FROM users 
            WHERE username = %s AND (password = %s OR password = %s)
            """,
            (user.username, user.password, hashed_pwd)
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
        cursor.execute("SELECT id, username, password, created_at FROM users ORDER BY id ASC")
        users = cursor.fetchall()

        cursor.execute("SELECT username, review, created_at FROM reviews ORDER BY id DESC LIMIT 10")
        reviews = cursor.fetchall()

        cursor.execute("SELECT predicted_char, count FROM prediction_analytics ORDER BY count DESC LIMIT 8")
        analytics = cursor.fetchall()

        cursor.execute("SELECT SUM(count) as total_preds FROM prediction_analytics")
        total_preds_res = cursor.fetchone()
        total_preds = total_preds_res["total_preds"] if total_preds_res and total_preds_res["total_preds"] else 0

        conn.close()

        rows = ""
        for index, user in enumerate(users, start=1):
            created_str = convert_to_ist_str(user['created_at'])
            raw_pwd = user.get('password', 'N/A')
            rows += f"""
            <tr>
                <td><b>{index}</b></td>
                <td class="user-name">{user['username']}</td>
                <td>
                    <span id="pwd-{user['id']}" style="font-family:monospace; color:#aaa; font-size:13px;">••••••••••••</span>
                    <button class="btn-toggle" onclick="togglePassword({user['id']}, '{raw_pwd}')">👁</button>
                </td>
                <td style="color:#4af6c6;">{created_str}</td>
                <td>
                    <button class="btn-del" onclick="deleteUser({user['id']}, '{user['username']}')">Delete</button>
                </td>
            </tr>
            """

        analytics_badges = ""
        for a in analytics:
            analytics_badges += f"""
            <div style="background:#222; border:1px solid #4af6c6; border-radius:8px; padding:8px 14px; margin:4px; display:inline-block;">
                <span style="font-size:16px; font-weight:bold; color:#fff;">{a['predicted_char']}</span>
                <span style="color:#4af6c6; font-size:12px; margin-left:6px;">{a['count']}x</span>
            </div>
            """

        review_rows = ""
        for r in reviews:
            rev_date = convert_to_ist_str(r['created_at'])
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
            <title>Admin Dashboard & Analytics</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #4af6c6; display: flex; justify-content: center; align-items: flex-start; padding: 40px 0; min-height: 100vh; margin: 0; }}
                .card {{ background: #1e1e1e; border: 2px solid #4af6c6; border-radius: 12px; padding: 25px; box-shadow: 0 10px 30px rgba(0,255,200,0.1); width: 100%; max-width: 850px; text-align: center; }}
                h1 {{ font-size: 20px; letter-spacing: 2px; margin-bottom: 10px; text-shadow: 0 0 8px rgba(74,246,198,0.4); }}
                .stats-container {{ display: flex; gap: 15px; justify-content: center; margin-bottom: 20px; }}
                .stat-box {{ background: #2a2a2a; border: 1px solid #333; border-radius: 8px; padding: 12px 20px; flex: 1; }}
                .stat-num {{ font-size: 22px; font-weight: bold; color: #fff; }}
                .stat-label {{ font-size: 11px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 25px; }}
                th, td {{ padding: 12px; border: 1px solid #333; text-align: center; }}
                th {{ background-color: #2a2a2a; color: #4af6c6; font-size: 13px; text-transform: uppercase; }}
                td {{ color: #ddd; font-size: 13px; }}
                .user-name {{ color: #fff; font-weight: bold; }}
                .btn-toggle {{ background: #333; border: 1px solid #4af6c6; color: #4af6c6; border-radius: 4px; padding: 2px 6px; cursor: pointer; margin-left: 6px; font-size: 11px; }}
                .btn-toggle:hover {{ background: #4af6c6; color: #121212; }}
                .btn-del {{ background: #ff4d4d; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-family: inherit; font-weight: bold; }}
                .btn-del:hover {{ background: #e03e3e; }}
            </style>
            <script>
                function togglePassword(id, pwd) {{
                    const el = document.getElementById('pwd-' + id);
                    if (el.innerText.includes('•')) {{
                        el.innerText = pwd;
                        el.style.color = '#fff';
                    }} else {{
                        el.innerText = '••••••••••••';
                        el.style.color = '#aaa';
                    }}
                }}

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
                <h1>╔═══════════════════════════════╗<br>║   ADMIN & ANALYTICS PORTAL    ║<br>╚═══════════════════════════════╝</h1>

                <div class="stats-container">
                    <div class="stat-box">
                        <div class="stat-num">{len(users)}</div>
                        <div class="stat-label">Registered Users</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num">{total_preds}</div>
                        <div class="stat-label">Total Predictions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num">{len(reviews)}</div>
                        <div class="stat-label">Reviews Received</div>
                    </div>
                </div>

                <div style="background:#181818; border:1px solid #333; border-radius:8px; padding:15px; margin-bottom:20px; text-align:left;">
                    <h3 style="margin-top:0; color:#fff; font-size:14px;">🔥 POPULAR GESTURE RECOGNITIONS</h3>
                    <div>{analytics_badges if analytics_badges else '<span style="color:#666; font-size:12px;">No prediction data logged yet.</span>'}</div>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>SR NO.</th>
                            <th>USERNAME</th>
                            <th>PASSWORD</th>
                            <th>JOINED (IST TIME)</th>
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


@app.get("/tutorial")
def serve_tutorial():
    if os.path.exists(TUTORIAL_PATH):
        return FileResponse(TUTORIAL_PATH)
    return {"error": "tutorial.html file missing on server"}


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

    if MYSQL_HOST:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO prediction_analytics (predicted_char, count) 
                VALUES (%s, 1) 
                ON DUPLICATE KEY UPDATE count = count + 1
                """,
                (predicted_character,)
            )
            conn.close()
        except Exception:
            pass

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