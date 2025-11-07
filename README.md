# Crime Rate Prediction & Analysis using K-Means Clustering Algorithm

## 🧾 Project Overview  
This Django-based web application enables uploading crime datasets (from :contentReference[oaicite:0]{index=0} of India or similar), performing data analysis, clustering via :contentReference[oaicite:1]{index=1}, and future prediction of crime trends.  
Goal: Visualise crime trends by state, cluster states based on crime/chargesheeting rates, and predict future crime rates.

## 🎯 Features  
- User login authentication (only registered users can upload/analyse)  
- Dataset upload (CSV format)  
- Data cleaning and sanitisation (skips bad lines)  
- Visualisations:  
  - Crime counts by state (bar chart)  
  - Rate vs chargesheet scatter plot  
  - K-Means clustering of states  
- Future prediction module (linear regression based)  
- Responsive UI with modern navbar, dark theme and full-screen graphs

## 🛠 Technology Stack  
- Backend: :contentReference[oaicite:2]{index=2} (Python)  
- Frontend: HTML, CSS, Bootstrap 5  
- Data Science Stack: :contentReference[oaicite:3]{index=3}, :contentReference[oaicite:4]{index=4}, :contentReference[oaicite:5]{index=5}  
- Version control: :contentReference[oaicite:6]{index=6} + :contentReference[oaicite:7]{index=7}  
- Dataset source: NCRB or analogous formatted dataset

## 📁 Repository Structure  
crime_project/
├── manage.py
├── crime_project/ – Django settings
├── core/ – main app (views, urls, templates, static)
├── data/ – uploaded datasets (CSV stored)
├── static/ – CSS, JS, images, graphs
└── templates/
├── login.html
├── index.html
├── upload.html
├── analysis.html
├── cluster.html
└── future.html

## 🚀 Setup & Run Locally  
1. Clone the repository  
   ```bash
   git clone https://github.com/sravanakkaladevi/CRIME-RATE-PREDICTION-ANALYSIS-USING-K-MEANS-CLUSTERING-ALGORITHM.git
Navigate to the project folder

cd CRIME-RATE-PREDICTION-ANALYSIS-USING-K-MEANS-CLUSTERING-ALGORITHM


Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Set up database & static files

python manage.py migrate
python manage.py collectstatic


Create a superuser for login

python manage.py createsuperuser


Use a valid email when prompted.

Run the server

python manage.py runserver


📄 Usage

After login: Upload CSV dataset via Upload tab

Navigate to Analysis to view visualisations

Use Cluster to input custom rate/charge values and get a cluster label

Use Future to input values and generate future crime rate prediction

Logout when done.

🎓 Academic Use / Submission

This project was developed for an academic course as a semester-end project. The logic and dataset should be used responsibly.
Note: The default admin username/password (if set for demo) must be changed in production.

🧬 Future Enhancements

Use a full regression model or ML pipeline for better prediction accuracy

Add user registration + roles (admin / regular user)

Chart enhancements (hover tooltips, filters by year or state)

Deploy on cloud (Heroku / Azure / AWS) with CI/CD

Add dataset downloader links for students.

📜 License

This project is distributed under the MIT License — feel free to use, adapt and extend.


---

If you like, I can generate a **ready-to-copy `requirements.txt`** listing all your current Python packages (based on your environment). Do you want that?
::contentReference[oaicite:8]{index=8}
Visit http://127.0.0.1:8000/ and log in to begin.
