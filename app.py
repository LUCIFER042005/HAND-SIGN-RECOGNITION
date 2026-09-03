import io
import json
import pickle
import hashlib
import os
import random
import secrets
from datetime import datetime, timezone, timedelta
import pymysql
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mediapipe as mp
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = FastAPI(title="Hand Sign Recognition API", version="1.0")

security = HTTPBasic()

# --- ADMIN CREDENTIALS ---
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme_local_pass")

# --- TIMEZONE CONFIGURATION (IST) ---
IST = timezone(timedelta(hours=5, minutes=30))


def convert_to_ist_str(dt_val):
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
DATA_PICKLE_PATH = os.path.join(BASE_DIR, "data.pickle")
STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB", "defaultdb")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 15747)) if os.getenv("MYSQL_PORT") else 15747

ALL_SUPPORTED_SIGNS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "N", "O", "P", "Q", "R", "U", "V", "W",
    "X", "Y", "DELETE", "SPACE"
]

GOLDEN_CENTROIDS = {}


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


def compute_golden_centroids():
    global GOLDEN_CENTROIDS
    if not os.path.exists(DATA_PICKLE_PATH):
        return
    try:
        with open(DATA_PICKLE_PATH, "rb") as f:
            base_dict = pickle.load(f)
            data = base_dict["data"]
            labels = base_dict["labels"]

        clusters = {}
        for d, l in zip(data, labels):
            clusters.setdefault(l, []).append(d)

        for lbl, samples in clusters.items():
            arr = np.array(samples)
            mean_vec = np.mean(arr, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            GOLDEN_CENTROIDS[lbl] = mean_vec

        print(f"Self-Cleaning AI: Loaded golden centroids for {len(GOLDEN_CENTROIDS)} signs.")
    except Exception as e:
        print(f"Error computing golden centroids: {e}")


def is_sample_clean(sign_label: str, landmarks: list[float], threshold: float = 0.80) -> tuple[bool, float]:
    global model, GOLDEN_CENTROIDS

    if len(landmarks) != 42:
        return False, 0.0

    cand_vec = np.array(landmarks)
    norm = np.linalg.norm(cand_vec)
    if norm > 0:
        cand_norm = cand_vec / norm
    else:
        return False, 0.0

    cos_sim = 1.0
    if sign_label in GOLDEN_CENTROIDS:
        target_vec = GOLDEN_CENTROIDS[sign_label]
        cos_sim = float(np.dot(target_vec, cand_norm))

    if cos_sim < threshold:
        return False, cos_sim

    return True, cos_sim


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
                admin_reply TEXT NULL,
                created_at DATETIME NOT NULL
            )
        """)
        try:
            cursor.execute("ALTER TABLE reviews ADD COLUMN admin_reply TEXT NULL")
        except Exception:
            pass

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_analytics (
                predicted_char VARCHAR(20) PRIMARY KEY,
                count INT DEFAULT 1
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hand_samples (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                sign_label VARCHAR(20) NOT NULL,
                landmarks JSON NOT NULL,
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


class DeleteOwnReview(BaseModel):
    username: str
    review_id: int


class AdminReply(BaseModel):
    review_id: int
    admin_reply: str


class SignContribution(BaseModel):
    username: str
    sign_label: str
    landmarks: list[float]


model = None
hands = None


@app.on_event("startup")
def load_resources():
    global model, hands
    init_db()
    compute_golden_centroids()

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


@app.get("/users/reviews/{username}")
def get_user_reviews_and_replies(username: str):
    if not MYSQL_HOST:
        return {"reviews": []}
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, review, admin_reply, created_at 
            FROM reviews 
            WHERE username = %s 
            ORDER BY id DESC
            """,
            (username,)
        )
        rows = cursor.fetchall()
        conn.close()
        for r in rows:
            r['created_at'] = convert_to_ist_str(r['created_at'])
        return {"reviews": rows}
    except Exception as e:
        return {"reviews": [], "error": str(e)}


@app.post("/users/delete-review")
def delete_own_review(data: DeleteOwnReview):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM reviews WHERE id = %s AND username = %s",
            (data.review_id, data.username)
        )
        affected = cursor.rowcount
        conn.close()
        if affected > 0:
            return {"success": True, "message": "Review deleted."}
        else:
            raise HTTPException(status_code=403, detail="Review not found or unauthorized.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/admin/reply-review")
def reply_to_review(data: AdminReply, admin: str = Depends(authenticate_admin)):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE reviews SET admin_reply = %s WHERE id = %s",
            (data.admin_reply, data.review_id)
        )
        affected = cursor.rowcount
        conn.close()
        if affected > 0:
            return {"success": True, "message": "Reply saved and sent to user!"}
        else:
            raise HTTPException(status_code=404, detail="Review not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.delete("/admin/reviews/{review_id}")
def delete_entire_review_by_admin(review_id: int, admin: str = Depends(authenticate_admin)):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reviews WHERE id = %s", (review_id,))
        affected = cursor.rowcount
        conn.close()
        if affected > 0:
            return {"success": True, "message": f"Review {review_id} deleted."}
        else:
            raise HTTPException(status_code=404, detail="Review not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.delete("/admin/reviews/{review_id}/reply")
def delete_admin_reply_only(review_id: int, admin: str = Depends(authenticate_admin)):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE reviews SET admin_reply = NULL WHERE id = %s", (review_id,))
        affected = cursor.rowcount
        conn.close()
        if affected > 0:
            return {"success": True, "message": "Admin reply removed."}
        else:
            raise HTTPException(status_code=404, detail="Review not found.")
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


@app.get("/api/contribute/random-signs")
def get_random_signs_to_contribute():
    sample_signs = random.sample(ALL_SUPPORTED_SIGNS, 3)
    return {"signs": sample_signs}


@app.post("/api/contribute-sign")
def submit_hand_sample(data: SignContribution):
    if len(data.landmarks) != 42:
        raise HTTPException(status_code=400, detail="Invalid landmark array length. Expected 42 floats.")

    is_valid, sim_score = is_sample_clean(data.sign_label, data.landmarks, threshold=0.80)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Sample rejected by Self-Cleaning AI. Match score: {sim_score * 100:.1f}% (Required: ≥80%). Please hold the exact sign clearly."
        )

    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database connection not configured.")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        created_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO hand_samples (username, sign_label, landmarks, created_at) 
            VALUES (%s, %s, %s, %s)
            """,
            (data.username, data.sign_label, json.dumps(data.landmarks), created_at)
        )
        conn.close()
        return {"success": True,
                "message": f"Sample for '{data.sign_label}' verified & saved! (Match: {sim_score * 100:.1f}%)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store sample: {str(e)}")


def retrain_model_pipeline():
    global model
    all_data = []
    all_labels = []

    if os.path.exists(DATA_PICKLE_PATH):
        try:
            with open(DATA_PICKLE_PATH, "rb") as f:
                base_dict = pickle.load(f)
                all_data.extend(base_dict["data"])
                all_labels.extend(base_dict["labels"])
        except Exception as e:
            print(f"Error loading base pickle: {e}")

    crowdsourced_accepted = 0
    crowdsourced_rejected = 0
    if MYSQL_HOST:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT sign_label, landmarks FROM hand_samples")
            rows = cursor.fetchall()
            conn.close()

            samples_by_sign = {}
            for r in rows:
                lbl = r["sign_label"]
                lms = json.loads(r["landmarks"])
                samples_by_sign.setdefault(lbl, []).append(lms)

            for lbl, lms_list in samples_by_sign.items():
                if len(lms_list) >= 4:
                    iso = IsolationForest(contamination=0.10, random_state=42)
                    preds = iso.fit_predict(lms_list)
                    for idx, pred in enumerate(preds):
                        if pred == 1:
                            all_data.append(lms_list[idx])
                            all_labels.append(lbl)
                            crowdsourced_accepted += 1
                        else:
                            crowdsourced_rejected += 1
                else:
                    for lms in lms_list:
                        all_data.append(lms)
                        all_labels.append(lbl)
                        crowdsourced_accepted += 1

        except Exception as e:
            print(f"Error filtering DB samples: {e}")

    if not all_data:
        return {"success": False, "message": "No training samples found."}

    X = np.asarray(all_data)
    y = np.asarray(all_labels)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf}, f)

    model = clf
    compute_golden_centroids()

    return {
        "success": True,
        "total_samples": len(all_data),
        "crowdsourced_accepted": crowdsourced_accepted,
        "crowdsourced_rejected": crowdsourced_rejected,
        "accuracy": round(acc * 100, 2)
    }


@app.post("/admin/retrain")
def trigger_retrain(background_tasks: BackgroundTasks, admin: str = Depends(authenticate_admin)):
    res = retrain_model_pipeline()
    return res


@app.post("/admin/clean-db")
def auto_purge_junk_db(admin: str = Depends(authenticate_admin)):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, sign_label, landmarks FROM hand_samples")
        rows = cursor.fetchall()

        purged_ids = []
        for r in rows:
            lms = json.loads(r["landmarks"])
            is_valid, _ = is_sample_clean(r["sign_label"], lms, threshold=0.80)
            if not is_valid:
                purged_ids.append(r["id"])

        if purged_ids:
            format_strings = ','.join(['%s'] * len(purged_ids))
            cursor.execute(f"DELETE FROM hand_samples WHERE id IN ({format_strings})", tuple(purged_ids))

        conn.close()
        return {
            "success": True,
            "scanned_total": len(rows),
            "purged_junk_count": len(purged_ids),
            "retained_valid_count": len(rows) - len(purged_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.delete("/admin/samples/{sample_id}")
def delete_contributed_sample(sample_id: int, admin: str = Depends(authenticate_admin)):
    if not MYSQL_HOST:
        raise HTTPException(status_code=500, detail="Database credentials missing on server.")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM hand_samples WHERE id = %s", (sample_id,))
        affected = cursor.rowcount
        conn.close()

        if affected > 0:
            return {"success": True, "message": f"Sample {sample_id} deleted."}
        else:
            raise HTTPException(status_code=404, detail="Sample not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


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

        cursor.execute("SELECT id, username, review, admin_reply, created_at FROM reviews ORDER BY id DESC LIMIT 20")
        reviews = cursor.fetchall()

        cursor.execute("SELECT predicted_char, count FROM prediction_analytics ORDER BY count DESC LIMIT 8")
        analytics = cursor.fetchall()

        cursor.execute("SELECT id, username, sign_label, landmarks, created_at FROM hand_samples ORDER BY id DESC")
        samples = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) as total_samples FROM hand_samples")
        samples_res = cursor.fetchone()
        total_samples = samples_res["total_samples"] if samples_res else 0

        conn.close()

        user_rows = ""
        for index, user in enumerate(users, start=1):
            created_str = convert_to_ist_str(user['created_at'])
            raw_pwd = user.get('password', 'N/A')
            user_rows += f"""
            <tr>
                <td><b>{index}</b></td>
                <td style="color:#fff; font-weight:600;">{user['username']}</td>
                <td>
                    <span id="pwd-{user['id']}" style="font-family:monospace; color:#64748b; font-size:13px;">••••••••••••</span>
                    <button class="btn-icon" onclick="togglePassword({user['id']}, '{raw_pwd}')">👁</button>
                </td>
                <td style="color:#94a3b8; font-size:12px;">{created_str}</td>
                <td>
                    <button class="btn-danger" onclick="deleteUser({user['id']}, '{user['username']}')">Delete</button>
                </td>
            </tr>
            """

        grouped_samples = {}
        for s in samples:
            lbl = s['sign_label']
            grouped_samples.setdefault(lbl, []).append(s)

        grouped_samples_html = ""
        if not grouped_samples:
            grouped_samples_html = '<div style="color:#64748b; padding:15px; font-size:13px;">No community samples recorded yet.</div>'
        else:
            for sign_lbl, s_list in sorted(grouped_samples.items()):
                sample_cards = ""
                for s in s_list:
                    s_date = convert_to_ist_str(s['created_at'])
                    lms_raw = s['landmarks'] if isinstance(s['landmarks'], str) else json.dumps(s['landmarks'])
                    sample_cards += f"""
                    <div style="background:#0d0f12; border:1px solid #262b32; padding:10px; text-align:center; min-width:130px; flex:1; max-width:160px;">
                        <canvas id="cvs-{s['id']}" width="70" height="70" style="background:#000; border:1px solid #2d333b; display:block; margin:0 auto 6px auto;"></canvas>
                        <script>drawSkeleton('cvs-{s['id']}', {lms_raw});</script>
                        <div style="font-size:11px; color:#fff; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">@{s['username']}</div>
                        <div style="font-size:10px; color:#64748b; margin-top:2px;">{s_date.split(',')[0]}</div>
                        <button class="btn-danger" style="margin-top:6px; padding:3px 8px; font-size:11px; width:100%;" onclick="deleteSample({s['id']})">🗑️ Delete</button>
                    </div>
                    """

                grouped_samples_html += f"""
                <details style="background:#13161a; border:1px solid #262b32; margin-bottom:10px;">
                    <summary style="padding:14px 16px; cursor:pointer; font-weight:600; display:flex; justify-content:space-between; align-items:center; list-style:none; outline:none; user-select:none;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span style="background:#162320; border:1px solid #22c55e; padding:3px 10px; font-weight:bold; font-size:14px; color:#22c55e;">{sign_lbl}</span>
                            <span style="color:#cbd5e1; font-size:14px;">Sign "{sign_lbl}"</span>
                        </div>
                        <span style="color:#94a3b8; font-size:12px; font-weight:600;">📁 {len(s_list)} sample(s) <span style="font-size:10px; margin-left:6px;">▼</span></span>
                    </summary>
                    <div style="padding:14px; border-top:1px solid #22262c; background:#101317; display:flex; flex-wrap:wrap; gap:10px;">
                        {sample_cards}
                    </div>
                </details>
                """

        review_rows = ""
        for r in reviews:
            rev_date = convert_to_ist_str(r['created_at'])
            curr_reply = r.get('admin_reply') or ""
            saved_reply_box = ""
            if curr_reply:
                saved_reply_box = f"""
                <div style="background:#0a0c0e; border-left:3px solid #22c55e; padding:8px 12px; margin-top:8px; font-size:12px; color:#4ade80; display:flex; justify-content:space-between; align-items:center;">
                    <div><b>Saved Reply:</b> "{curr_reply}"</div>
                    <button class="btn-danger" style="padding:2px 6px; font-size:10px;" onclick="deleteAdminReplyOnly({r['id']})">🗑️ Delete Reply</button>
                </div>
                """

            review_rows += f"""
            <div style="background:#13161a; border:1px solid #262b32; padding:16px; margin-bottom:14px; text-align:left;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:13px; color:#22c55e; font-weight:700;">@{r['username']}</span>
                        <span style="color:#64748b; font-size:12px; margin-left:8px;">{rev_date}</span>
                    </div>
                    <button class="btn-danger" style="padding:3px 8px; font-size:11px;" onclick="deleteEntireReview({r['id']})">🗑️ Delete User Review</button>
                </div>
                <div style="font-size:14px; color:#cbd5e1; margin-top:8px; line-height:1.4;">"{r['review']}"</div>
                {saved_reply_box}

                <div style="margin-top:12px; padding-top:10px; border-top:1px solid #22262c;">
                    <div style="font-size:11px; color:#94a3b8; margin-bottom:6px; font-weight:600;">{('EDIT / SEND REPLY' if curr_reply else 'ADMIN REPLY')}:</div>
                    <div style="display:flex; gap:6px;">
                        <input type="text" id="reply-input-{r['id']}" value="{curr_reply}" placeholder="Type reply to @{r['username']}..." style="flex:1; padding:8px 12px; font-size:12px; background:#0d0f12; border:1px solid #2d333b; color:#fff; outline:none;">
                        <button class="btn-action" style="padding:8px 16px; font-size:12px;" onclick="sendAdminReply({r['id']})">Send Reply</button>
                    </div>
                </div>
            </div>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Admin Dashboard & Self-Cleaning AI</title>
            <style>
                * {{ box-sizing: border-box; border-radius: 0 !important; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    background-color: #121417;
                    color: #f8fafc;
                    display: flex;
                    justify-content: center;
                    align-items: flex-start;
                    padding: 40px 15px;
                    min-height: 100vh;
                    margin: 0;
                }}
                .card {{
                    background: #181b20;
                    border: 1px solid #262b32;
                    padding: 30px;
                    box-shadow: 0 12px 35px rgba(0,0,0,0.45);
                    width: 100%;
                    max-width: 900px;
                    text-align: center;
                }}
                h1 {{
                    font-size: 22px;
                    font-weight: 700;
                    letter-spacing: 0.5px;
                    margin-bottom: 20px;
                    color: #ffffff;
                }}
                .stats-container {{
                    display: flex;
                    gap: 15px;
                    justify-content: center;
                    margin-bottom: 24px;
                }}
                .stat-box {{
                    background: #13161a;
                    border: 1px solid #262b32;
                    padding: 16px 20px;
                    flex: 1;
                }}
                .stat-num {{
                    font-size: 26px;
                    font-weight: 700;
                    color: #22c55e;
                }}
                .stat-label {{
                    font-size: 11px;
                    color: #94a3b8;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-top: 4px;
                    font-weight: 600;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 12px;
                    margin-bottom: 28px;
                }}
                th, td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #22262c;
                    text-align: center;
                }}
                th {{
                    background-color: #13161a;
                    color: #94a3b8;
                    font-size: 12px;
                    text-transform: uppercase;
                    font-weight: 600;
                    letter-spacing: 0.5px;
                }}
                td {{
                    color: #cbd5e1;
                    font-size: 13px;
                }}
                .btn-icon {{
                    background: #1e2229;
                    border: 1px solid #333942;
                    color: #94a3b8;
                    padding: 3px 8px;
                    cursor: pointer;
                    font-size: 12px;
                    margin-left: 6px;
                }}
                .btn-icon:hover {{
                    background: #282e38;
                    color: #fff;
                }}
                .btn-danger {{
                    background: #2a1618;
                    border: 1px solid #ef4444;
                    color: #f87171;
                    padding: 5px 12px;
                    cursor: pointer;
                    font-weight: 600;
                    font-size: 12px;
                }}
                .btn-danger:hover {{
                    background: #ef4444;
                    color: #fff;
                }}
                .btn-action {{
                    background: #22c55e;
                    color: #ffffff;
                    border: none;
                    padding: 10px 22px;
                    font-weight: 600;
                    cursor: pointer;
                    font-size: 14px;
                    margin: 4px;
                    transition: 0.2s;
                }}
                .btn-action:hover {{
                    background: #16a34a;
                }}
                .btn-clean {{
                    background: #1e2229;
                    color: #f8fafc;
                    border: 1px solid #3b424e;
                    padding: 10px 22px;
                    font-weight: 600;
                    cursor: pointer;
                    font-size: 14px;
                    margin: 4px;
                    transition: 0.2s;
                }}
                .btn-clean:hover {{
                    background: #262c36;
                }}
                .section-header {{
                    color: #ffffff;
                    font-size: 15px;
                    font-weight: 600;
                    border-top: 1px solid #262b32;
                    padding-top: 22px;
                    text-align: left;
                    margin-bottom: 12px;
                }}
                summary::-webkit-details-marker {{
                    display: none;
                }}
                details[open] summary {{
                    background: #181c22;
                }}
            </style>
            <script>
                function drawSkeleton(canvasId, lms) {{
                    setTimeout(() => {{
                        const cvs = document.getElementById(canvasId);
                        if (!cvs || !lms || lms.length !== 42) return;
                        const ctx = cvs.getContext('2d');
                        ctx.clearRect(0, 0, cvs.width, cvs.height);

                        let pts = [];
                        let minX = 999, maxX = -999, minY = 999, maxY = -999;
                        for (let i = 0; i < lms.length; i += 2) {{
                            let x = lms[i], y = lms[i+1];
                            pts.push({{x, y}});
                            if (x < minX) minX = x; if (x > maxX) maxX = x;
                            if (y < minY) minY = y; if (y > maxY) maxY = y;
                        }}
                        let w = (maxX - minX) || 1, h = (maxY - minY) || 1;
                        let pad = 6;

                        ctx.strokeStyle = '#22c55e';
                        ctx.fillStyle = '#ef4444';
                        ctx.lineWidth = 1.5;

                        const connections = [
                            [0,1],[1,2],[2,3],[3,4],
                            [0,5],[5,6],[6,7],[7,8],
                            [5,9],[9,10],[10,11],[11,12],
                            [9,13],[13,14],[14,15],[15,16],
                            [13,17],[17,18],[18,19],[19,20],[0,17]
                        ];

                        connections.forEach(([i, j]) => {{
                            let x1 = pad + ((pts[i].x - minX) / w) * (cvs.width - pad*2);
                            let y1 = pad + ((pts[i].y - minY) / h) * (cvs.height - pad*2);
                            let x2 = pad + ((pts[j].x - minX) / w) * (cvs.width - pad*2);
                            let y2 = pad + ((pts[j].y - minY) / h) * (cvs.height - pad*2);
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2, y2);
                            ctx.stroke();
                        }});

                        pts.forEach(p => {{
                            let px = pad + ((p.x - minX) / w) * (cvs.width - pad*2);
                            let py = pad + ((p.y - minY) / h) * (cvs.height - pad*2);
                            ctx.beginPath();
                            ctx.arc(px, py, 1.8, 0, 2 * Math.PI);
                            ctx.fill();
                        }});
                    }}, 50);
                }}

                function togglePassword(id, pwd) {{
                    const el = document.getElementById('pwd-' + id);
                    if (el.innerText.includes('•')) {{
                        el.innerText = pwd;
                        el.style.color = '#f8fafc';
                    }} else {{
                        el.innerText = '••••••••••••';
                        el.style.color = '#64748b';
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

                async function deleteSample(id) {{
                    if (confirm("Delete this sample?")) {{
                        const res = await fetch('/admin/samples/' + id, {{ method: 'DELETE' }});
                        if (res.ok) {{
                            alert("Sample removed.");
                            window.location.reload();
                        }} else {{
                            alert("Failed to delete sample.");
                        }}
                    }}
                }}

                async function deleteEntireReview(reviewId) {{
                    if (confirm("Permanently delete this user review?")) {{
                        try {{
                            const res = await fetch('/admin/reviews/' + reviewId, {{ method: 'DELETE' }});
                            if (res.ok) {{
                                alert("Review deleted.");
                                window.location.reload();
                            }} else {{
                                alert("Failed to delete review.");
                            }}
                        }} catch (e) {{
                            alert("Network error deleting review.");
                        }}
                    }}
                }}

                async function deleteAdminReplyOnly(reviewId) {{
                    if (confirm("Delete your reply for this review?")) {{
                        try {{
                            const res = await fetch('/admin/reviews/' + reviewId + '/reply', {{ method: 'DELETE' }});
                            if (res.ok) {{
                                alert("Admin reply removed.");
                                window.location.reload();
                            }} else {{
                                alert("Failed to remove reply.");
                            }}
                        }} catch (e) {{
                            alert("Network error removing reply.");
                        }}
                    }}
                }}

                async function sendAdminReply(reviewId) {{
                    const input = document.getElementById('reply-input-' + reviewId);
                    const replyText = input.value.trim();
                    if (!replyText) {{
                        alert("Please type a reply message.");
                        return;
                    }}
                    try {{
                        const res = await fetch('/admin/reply-review', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{ review_id: reviewId, admin_reply: replyText }})
                        }});
                        if (res.ok) {{
                            alert("Reply successfully saved and sent to user!");
                            window.location.reload();
                        }} else {{
                            alert("Failed to send reply.");
                        }}
                    }} catch(e) {{
                        alert("Network error sending reply.");
                    }}
                }}

                async function cleanDatabase() {{
                    if (confirm("Run Self-Cleaning AI Janitor on all stored samples?")) {{
                        try {{
                            const res = await fetch('/admin/clean-db', {{ method: 'POST' }});
                            const data = await res.json();
                            alert("Janitor complete!\\nScanned: " + data.scanned_total + " samples\\nAuto-Purged Outliers: " + data.purged_junk_count + "\\nRetained Valid: " + data.retained_valid_count);
                            window.location.reload();
                        }} catch(e) {{
                            alert("Error running DB Janitor.");
                        }}
                    }}
                }}

                async function retrainModel() {{
                    const btn = document.getElementById('retrainBtn');
                    btn.innerText = "⏳ Retraining Model...";
                    btn.disabled = true;
                    try {{
                        const res = await fetch('/admin/retrain', {{ method: 'POST' }});
                        const data = await res.json();
                        if (data.success) {{
                            alert("Model successfully upgraded!\\nTotal Training Samples: " + data.total_samples + "\\nAuto-Accepted Community Samples: " + data.crowdsourced_accepted + "\\nAuto-Discarded Outlier Junk: " + data.crowdsourced_rejected + "\\nNew Accuracy: " + data.accuracy + "%");
                            window.location.reload();
                        }} else {{
                            alert("Retraining failed: " + data.message);
                        }}
                    }} catch (e) {{
                        alert("Error contacting retrain endpoint.");
                    }} finally {{
                        btn.innerText = "🚀 Retrain Model on Community Data";
                        btn.disabled = false;
                    }}
                }}
            </script>
        </head>
        <body>
            <div class="card">
                <h1>Admin Control & Auto-Trainer Portal</h1>

                <div class="stats-container">
                    <div class="stat-box">
                        <div class="stat-num">{len(users)}</div>
                        <div class="stat-label">Registered Users</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num">{total_samples}</div>
                        <div class="stat-label">Contributed Samples</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num">{len(reviews)}</div>
                        <div class="stat-label">Reviews Received</div>
                    </div>
                </div>

                <div style="margin-bottom: 24px;">
                    <button id="retrainBtn" class="btn-action" onclick="retrainModel()">🚀 Retrain Model on Community Data</button>
                    <button class="btn-clean" onclick="cleanDatabase()">🧹 Run AI DB Janitor</button>
                </div>

                <div class="section-header">📂 Contributed Samples (Dropdown / Expandable)</div>
                {grouped_samples_html}

                <div class="section-header">👥 Registered Community Members</div>
                <table>
                    <thead>
                        <tr>
                            <th>SR NO.</th>
                            <th>USERNAME</th>
                            <th>PASSWORD</th>
                            <th>JOINED (IST)</th>
                            <th>ACTION</th>
                        </tr>
                    </thead>
                    <tbody>
                        {user_rows}
                    </tbody>
                </table>

                <div class="section-header">💬 User Reviews & Admin Reply System</div>
                {review_rows if review_rows else '<div style="color:#64748b; font-size:13px;">No reviews submitted yet.</div>'}
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
    return {"status": "online", "model_loaded": model is not None, "golden_centroids_loaded": len(GOLDEN_CENTROIDS)}


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
            "raw_features": [],
            "message": "No hand landmarks detected",
        }

    hand_landmarks = results.multi_hand_landmarks[0]

    # --- HANDEDNESS & LEFT-HAND MIRRORING ---
    # Because frontend mirrors the feed (scaleX(-1)), MediaPipe's raw "Right" label corresponds
    # to the user's physical Left hand, and "Left" corresponds to the user's physical Right hand.
    is_physical_left_hand = False
    detected_hand_label = "Right"
    if results.multi_handedness and len(results.multi_handedness) > 0:
        raw_label = results.multi_handedness[0].classification[0].label
        if raw_label == "Right":
            is_physical_left_hand = True
            detected_hand_label = "Left"
        else:
            detected_hand_label = "Right"

    x_ = []
    y_ = []

    # Extract coordinates; mirror horizontal axis if physical left hand was presented
    for i in range(len(hand_landmarks.landmark)):
        raw_x = hand_landmarks.landmark[i].x
        lm_x = (1.0 - raw_x) if is_physical_left_hand else raw_x
        lm_y = hand_landmarks.landmark[i].y
        x_.append(lm_x)
        y_.append(lm_y)

    min_x, max_x = min(x_), max(x_)
    min_y, max_y = min(y_), max(y_)
    box_w = max(max_x - min_x, 1e-6)
    box_h = max(max_y - min_y, 1e-6)

    data_aux = []
    for x, y in zip(x_, y_):
        data_aux.append((x - min_x) / box_w)
        data_aux.append((y - min_y) / box_h)

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
        "detected_hand": detected_hand_label,
        "raw_features": data_aux,
        "bbox": {
            "x_min": min_x,
            "y_min": min_y,
            "x_max": max_x,
            "y_max": max_y,
        },
    }