Control your computer's mouse and keyboard hands-free! This project uses real-time head pose estimation for cursor movement and facial expression recognition for actions like clicking and scrolling.

*(I highly recommend you record a 10-second GIF of it working and put it here. It's the most important part.)*

---

## ✨ Features

* **Mouse Control:** Move your head to control the cursor's position on the screen.
* **Facial Gestures:** Perform actions using expressions.
    * **Left Click:** Tilt Head left
    * **Right Click:** Right Wink
* **Scrolling:**
    * **Vertical Scroll:** Nod Head Up/Down
    * **Horizontal Scroll:** Head Left/Right
* **Real-time & Responsive:** All processing is done locally in real-time.

---

## 🛠️ Tech Stack

* **Core:** Python
* **Computer Vision:** OpenCV
* **Facial Landmark Detection:** [**IMPORTANT: Specify if you used MediaPipe or dlib here**]
* **OS Automation:** PyAutoGUI
* **Data Handling:** NumPy

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.12.9+
* A webcam

### 2. Installation

```bash
# 1. Clone the repository
git clone [https://github.com/Usaid786467/Facial-OS-Navigator.git](https://github.com/Usaid786467/Facial-OS-Navigator.git)
cd Facial-OS-Navigator

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install the required dependencies
pip install -r requirements.txt
