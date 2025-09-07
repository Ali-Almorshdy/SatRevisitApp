# SatRevisitApp

A **Streamlit-based web application** for analyzing satellite revisit times.  
The app provides two modes:  

1. **Manual Mode** – Calculate satellite passes and revisit times for **user-defined orbital elements** across multiple ground locations.  
2. **Optimization Mode (GA)** – Use a **Genetic Algorithm** to design satellite constellations that **minimize the average revisit time** over selected locations.  

This tool is designed for **space engineers, mission analysts, and researchers** working on Earth observation, communications, and coverage optimization problems.  

---

## ✨ Features

- Analyze **multiple satellites** and **multiple ground locations** in one run.  
- Compute access passes, revisit times, and global statistics.  
- **Manual Mode:**  
  - Direct calculation of satellite passes.  
  - Visualizations of access windows.  
- **GA Optimization Mode:**  
  - Evolutionary optimization of constellations.  
  - Minimize **average revisit time** across all locations.  
  - Progress tracking with fitness values per generation.  

---

## 📸 Screenshots  

<img width="1366" height="725" alt="M1" src="https://github.com/user-attachments/assets/2fd1a019-fc77-49f8-8b4b-e286530b350b" />

<img width="1361" height="574" alt="M2" src="https://github.com/user-attachments/assets/26617d18-4984-41fa-9efa-5662013634e7" />
<img width="1366" height="550" alt="m31" src="https://github.com/user-attachments/assets/52430ad8-34d7-41ab-bdb0-fce22b14533b" />

<img width="1365" height="593" alt="m3" src="https://github.com/user-attachments/assets/8d4e9c22-af30-46a1-a142-ee2f18e20962" />

## 🚀 How to Run

You can run SatRevisitApp locally or access it online.  

### Run Locally  
```bash
git clone https://github.com/yourusername/SatRevisitApp.git
pip install -r requirements.txt
streamlit run Revisit_app.py
```
### 🌐 Try it Online

You can access the app directly here:  
👉 [SatRevisitApp Online](https://ali-almorshdy-satrevisitapp-revisit-app-rvf9ln.streamlit.app/)

