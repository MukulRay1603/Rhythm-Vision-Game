# RhythmVision: A Real-Time Face-Driven Rhythm Interaction Game  
A lightweight rhythm game controlled entirely through head movement, built using MediaPipe Face Detection and custom motion tracking.

---

## 🎮 Overview  
RhythmVision is a real-time, webcam-based rhythm mini-game inspired by movement-centric titles like *Beat Saber*. Instead of handheld controllers, the player uses simple head motions—left, right, leaning in, or leaning back—to respond to visual prompts.

This project explores:  
- Real-time computer vision  
- MediaPipe face detection  
- Simple centroid-based tracking  
- Motion interpretation from bounding boxes  
- Interactive feedback and UI overlays  
- Visualization of facial motion trajectories  

No additional ML models are used—just face detection and lightweight tracking.

---

## 🧠 How the System Works  

### 1. **Face Detection (MediaPipe Tasks)**  
Each webcam frame is processed using MediaPipe’s BlazeFace-based short-range Face Detection model.

### 2. **Centroid Tracking**  
From each bounding box, the system computes the centroid `(cx, cy)`.  
A simple distance-based matching approach assigns a persistent ID to the face each frame.  
This smooths movement signals and reduces jitter compared to raw detection.

### 3. **Motion Extraction**  
Two motion metrics are derived:  
- **dx** → horizontal displacement (left/right)  
- **dA** → bounding-box area change (lean in/out)  

### 4. **Gameplay Logic**  
The game:  
- Generates random prompts every few seconds  
- Detects whether the user's movement matches  
- Awards XP, combos, and level-ups  
- Renders ripple and text animations  
- Logs all activity to `face_logs.csv`

### 5. **Trajectory Visualization**  
Using `plot_trajectories.py`, logged centroids are plotted to show the motion path of the face across the session.

---

## ⚙️ Features  
- Real-time face detection (MediaPipe)  
- Custom centroid-based face tracking  
- Stable face ID assignment  
- Motion-driven interactive gameplay  
- XP bar, combo counter, and animations  
- Auto logging of face motion  
- Trajectory visualization (Matplotlib)  

---

## 🖥️ System Requirements  
- Python **3.8+**  
- A working **webcam**  
- Windows/macOS/Linux (CPU only)  

---

# 🧩 Installation & Setup

Below is a clean, step-by-step setup that works on ANY machine.

---

## 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

---

## 2️⃣ **Create & Activate a Virtual Environment**

### **Windows (PowerShell)**
```bash
python -m venv venv
venv\Scripts\activate
```

### **macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

If activated successfully, your terminal should show:
```
(venv)
```

---

## 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

## 4️⃣ **Run the Rhythm Game**
This launches the webcam and starts the prompt-based motion game.

```bash
python main.py
```

**Gameplay Instructions:**  
- Follow the on-screen prompts  
- Move your head **left / right**  
- Lean **in / back**  
- Earn XP + combos for correct moves  

All movement logs will be written automatically to:
```
face_logs.csv
```

---

## 5️⃣ **Visualize Face Trajectories**
After playing the game, run:

```bash
python plot_trajectories.py
```

This generates a plot showing your face movement path over time.

---

# 📂 Project Structure  
```
.
├── main.py                  # Real-time rhythm interaction game
├── plot_trajectories.py     # Visualizer for logged facial motion
├── face_logs.csv            # Auto-generated during gameplay
├── requirements.txt
└── README.md
```

---

# 🔧 Tech Stack  
- **Python 3.8+**  
- **MediaPipe Tasks (Face Detection)**  
- **OpenCV**  
- **NumPy**  
- **Pandas**  
- **Matplotlib**  

---

# 🙌 Acknowledgements  
- MediaPipe by Google — for fast, lightweight face detection  
- Beat Saber — inspiration for movement-driven rhythm gameplay  
- OpenCV — for rendering and real-time UI overlays  

---

# 📜 License  
This project is free to use for learning, demos, and personal projects.
