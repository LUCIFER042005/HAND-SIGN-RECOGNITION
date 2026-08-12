# 🖐️ Hand Sign Recognition System

An end-to-end Machine Learning web application and REST API that detects and classifies hand signs in real-time. Built using **MediaPipe**, **OpenCV**, **Scikit-Learn**, **FastAPI**, and **MySQL**, this system enables custom gesture dataset collection, landmark feature extraction, model training, and web deployment.

---

## 🚀 Live Demo & API Documentation

- **Live Application:** [hand-sign-recognition-988z.onrender.com](https://hand-sign-recognition-988z.onrender.com)
- **Interactive API Docs (Swagger):** [hand-sign-recognition-988z.onrender.com/docs](https://hand-sign-recognition-988z.onrender.com/docs)

---

## ✨ Features

- **Custom Dataset Collector:** Interactive CLI tool (`collect_imgs.py`) with headless support for capturing webcam images for custom alphabet signs or gestures.
- **Landmark Feature Extraction:** Uses Google's **MediaPipe Hands** solution to extract 21 key 3D hand landmark coordinates.
- **Machine Learning Classification:** High-accuracy gesture classification trained with a **Random Forest Classifier** (`train_classifier.py`).
- **FastAPI REST Service:** Lightweight backend endpoints for real-time predictions (`/predict`), user authentication (`/register`, `/login`), and user reviews.
- **Admin Dashboard:** Built-in dashboard (`/admin/users`) with HTTP Basic Authentication, user management, and real-time IST timestamp tracking.
- **MySQL Database Integration:** Persistent user storage, password hashing (SHA-256), and review management.

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.10+
- **Computer Vision & ML:** MediaPipe, OpenCV, Scikit-Learn, NumPy, Pillow
- **Backend Framework:** FastAPI, Uvicorn, PyMySQL, Pydantic
- **Frontend:** HTML5, CSS3, JavaScript (Webcam API & Canvas integration)
- **Database:** MySQL
- **Deployment Platform:** Render

---

## 📂 Project Structure

```text
HAND-SIGN-RECOGNITION/
│
├── data/                    # Dataset directory (contains subfolders for each sign)
├── app.py                   # FastAPI application server & REST endpoints
├── collect_imgs.py          # Script to collect image samples via webcam
├── create_dataset.py        # Extracts hand landmarks and creates data.pickle
├── train_classifier.py      # Trains the Random Forest model and exports model.p
├── index.html               # Frontend interface for live camera interaction
├── requirements.txt         # Project dependencies
├── data.pickle              # Extracted landmark dataset file
└── model.p                  # Serialized trained machine learning model
