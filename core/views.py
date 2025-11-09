from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

def index(request):
    if not request.user.is_authenticated:
        return redirect('/')
    return render(request, 'index.html')


# ---------------- LOGIN PAGE ----------------
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # go to welcome page after login
        else:
            messages.error(request, "Invalid username or password.")
    return render(request, "login.html")


# ---------------- LOGOUT ----------------
def logout_view(request):
    logout(request)
    request.session.flush()  # 🧹 clear dataset session
    messages.info(request, "Logged out successfully.")
    return redirect('/')

# ---------------- HOME PAGE ----------------
@login_required(login_url='/')
def index(request):
    return render(request, "index.html")
# ---------------- UPLOAD PAGE ----------------
@login_required(login_url='/')
def upload_dataset(request):
    import pandas as pd
    import os
    from django.conf import settings

    context = {}
    data_dir = os.path.join(settings.BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "NCRB_Table_1A.1.csv")

    # ---------------- STEP 1: Handle File Upload ----------------
    if request.method == "POST" and request.FILES.get("file"):
        file = request.FILES["file"]
        try:
            df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
            df.to_csv(data_path, index=False)
            request.session["dataset_uploaded"] = True  # Save session
            context["msg"] = f"✅ Uploaded {file.name} successfully!"
        except Exception as e:
            context["msg"] = f"❌ Error reading file: {e}"

    # ---------------- STEP 2: Load Existing Dataset ----------------
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)

            # Populate dropdown lists safely (limit long lists)
        
            context["cities"] = sorted(df["City"].dropna().unique().tolist()[:100])
            context["crimes"] = sorted(df["Crime Description"].dropna().unique().tolist()[:100])
            context["genders"] = sorted(df["Victim Gender"].dropna().unique().tolist()[:50])
            context["weapons"] = sorted(df["Weapon Used"].dropna().unique().tolist()[:50])

            # Apply filters
            report = request.GET.get("report")
            city = request.GET.get("city")
            crime = request.GET.get("crime")
            gender = request.GET.get("gender")
            weapon = request.GET.get("weapon")

            filtered_df = df.copy()
            if report:
                filtered_df = filtered_df[filtered_df["Report Number"].astype(str) == str(report)]
            if city:
                filtered_df = filtered_df[filtered_df["City"] == city]
            if crime:
                filtered_df = filtered_df[filtered_df["Crime Description"] == crime]
            if gender:
                filtered_df = filtered_df[filtered_df["Victim Gender"] == gender]
            if weapon:
                filtered_df = filtered_df[filtered_df["Weapon Used"] == weapon]

            # Preview first 50 rows
            context["rows"] = filtered_df.head(50).to_html(classes="table table-striped table-sm", index=False)

        except Exception as e:
            context["msg"] = f"⚠️ Could not load dataset: {e}"

    # ---------------- STEP 3: No Dataset Yet ----------------
    elif not request.session.get("dataset_uploaded"):
        context["msg"] = "⚠️ Please upload a dataset to begin."

    return render(request, "upload.html", context)

# ---------------- ANALYSIS PAGE ----------------
@login_required(login_url='/login/')
def analyze_data(request):
    csv_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")
    if not os.path.exists(csv_path):
        messages.warning(request, "⚠️ Please upload the dataset first.")
        return redirect("/upload/")

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except Exception as e:
        messages.error(request, f"Error reading dataset: {e}")
        return redirect("/upload/")

    graphs_dir = os.path.join(settings.BASE_DIR, "static", "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Take smaller sample if data too large
    sample_df = df.sample(min(len(df), 20000), random_state=42)

    # 1️⃣ Top 10 Cities by Crime Count
    if "City" in sample_df.columns:
        plt.figure(figsize=(8, 4))
        sample_df["City"].fillna("Unknown").value_counts().head(10).plot(kind="bar", color="steelblue")
        plt.title("Top 10 Cities by Reported Crimes")
        plt.xlabel("City")
        plt.ylabel("Number of Crimes")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "top10_cities.png"))
        plt.close()

    # 2️⃣ Crimes by Victim Gender
    if "Victim Gender" in sample_df.columns:
        plt.figure(figsize=(4, 4))
        sample_df["Victim Gender"].fillna("Unknown").value_counts().plot(kind="pie", autopct="%1.1f%%")
        plt.title("Crimes by Victim Gender")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "gender_pie.png"))
        plt.close()

    # 3️⃣ Top Weapons Used
    if "Weapon Used" in sample_df.columns:
        plt.figure(figsize=(6, 4))
        sample_df["Weapon Used"].fillna("Unknown").value_counts().head(10).plot(kind="barh", color="firebrick")
        plt.title("Top 10 Weapons Used in Crimes")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "weapons.png"))
        plt.close()

    # 4️⃣ Crime Occurrence Time
    if "Time of Occurrence" in sample_df.columns:
        plt.figure(figsize=(8, 4))
        sample_df["Time of Occurrence"].astype(str).fillna("Unknown").value_counts().head(20).plot(kind="bar", color="orange")
        plt.title("Crimes by Time of Day (Top 20)")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "time_distribution.png"))
        plt.close()

    # Prepare preview table
    table_html = df.head(200).to_html(classes="table table-striped", index=False)

    context = {
        "table": table_html,
        "graphs": {
            "cities": "/static/graphs/top10_cities.png",
            "gender": "/static/graphs/gender_pie.png",
            "weapons": "/static/graphs/weapons.png",
            "time": "/static/graphs/time_distribution.png",
        },
    }
    return render(request, "analysis.html", context)
# ---------------- CLUSTER PREDICTION (with Elbow Method) ----------------
@login_required(login_url='/login/')
def cluster_prediction(request):
    result, cluster_plot = None, None
    csv_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")
    elbow_path = os.path.join(settings.BASE_DIR, "static", "graphs", "elbow_method.png")
    os.makedirs(os.path.dirname(elbow_path), exist_ok=True)

    if not os.path.exists(csv_path):
        messages.warning(request, "⚠️ Please upload the dataset first.")
        return redirect("/upload/")

    # Safe CSV load
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", dtype=str)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df = df.dropna(subset=["City", "Crime Description"])
    except Exception as e:
        messages.error(request, f"❌ Error reading dataset: {e}")
        return redirect("/upload/")

    # Combine columns for clustering
    df["combined"] = (
        df["City"].astype(str)
        + " " + df["Crime Description"].astype(str)
        + " " + df["Weapon Used"].astype(str)
        + " " + df["Victim Gender"].astype(str)
    )

    # Group text by city
    grouped = df.groupby("City")["combined"].apply(lambda x: " ".join(x)).reset_index()

    # Vectorize
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans

    vec = CountVectorizer(max_features=2000, stop_words="english")
    X = vec.fit_transform(grouped["combined"])

    # --- Elbow Method ---
    distortions = []
    K = range(1, 8)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        distortions.append(km.inertia_)

    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False

    plt.figure(figsize=(6, 4))
    if SEABORN_AVAILABLE:
        sns.lineplot(x=list(K), y=distortions, marker='o', color='#00eaff')
    else:
        plt.plot(list(K), distortions, marker='o', color='#00eaff')
    plt.title("Elbow Method - Finding Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(elbow_path)
    plt.close()

    # K-Means Clustering
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    grouped["cluster"] = km.fit_predict(X)

    # --- FIXED: Cluster labeling based on crime volume ---
    # Count number of crimes per cluster
    cluster_counts = grouped.groupby("cluster")["combined"].apply(lambda x: len(x))
    # Sort clusters by count: smallest → largest
    ordered_clusters = cluster_counts.sort_values().index.tolist()
    label_map = {
        ordered_clusters[0]: "Low Crime Zone",
        ordered_clusters[1]: "Medium Crime Zone",
        ordered_clusters[2]: "High Crime Zone",
    }

    if request.method == "POST":
        city = request.POST.get("city")
        if city not in grouped["City"].values:
            result = f"⚠️ City '{city}' not found in dataset."
        else:
            cl = int(grouped.loc[grouped["City"] == city, "cluster"].values[0])
            result = f"{city} is categorized as: {label_map.get(cl, 'Cluster')}"

            # Plot clusters
            svd = TruncatedSVD(n_components=2, random_state=42)
            coords = svd.fit_transform(X)
            plt.figure(figsize=(6, 4))
            colors = ["#00eaff", "#ffb347", "#a569bd"]
            for c in range(3):
                plt.scatter(coords[grouped["cluster"] == c, 0],
                            coords[grouped["cluster"] == c, 1],
                            label=label_map[c], alpha=0.6)
            idx = grouped.index[grouped["City"] == city][0]
            plt.scatter(coords[idx, 0], coords[idx, 1], s=120,
                        edgecolors="black", facecolors="none", linewidths=2)
            plt.title(f"Crime Clusters (highlighted: {city})")
            plt.legend()
            plt.tight_layout()
            cluster_plot_path = os.path.join(settings.BASE_DIR, "static", "graphs", "cluster_cities.png")
            plt.savefig(cluster_plot_path)
            plt.close()
            cluster_plot = "/static/graphs/cluster_cities.png"

    cities = sorted(df["City"].dropna().unique().tolist())
    crimes = sorted(df["Crime Description"].dropna().unique().tolist()[:200])
    weapons = sorted(df["Weapon Used"].dropna().unique().tolist()[:50]) if "Weapon Used" in df.columns else []
    genders = sorted(df["Victim Gender"].dropna().unique().tolist()[:50]) if "Victim Gender" in df.columns else []

    return render(
        request,
        "cluster.html",
        {
            "result": result,
            "cluster_plot": cluster_plot,
            "elbow_graph": "/static/graphs/elbow_method.png",
            "cities": cities,
            "crimes": crimes,
            "weapons": weapons,
            "genders": genders,
        },
    )

# ---------------- FUTURE PREDICTION WITH NORMALIZED TREND-BASED RISK ----------------
@login_required(login_url='/login/')
def future_prediction(request):
    result, line_plot, risk_level, risk_color = None, None, None, None
    csv_path = os.path.join(settings.BASE_DIR, "data", "NCRB_Table_1A.1.csv")

    # Step 1: Ensure dataset exists
    if not os.path.exists(csv_path):
        messages.warning(request, "⚠️ Please upload the dataset first.")
        return redirect("/upload/")

    # Step 2: Load dataset safely
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", dtype=str)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    except Exception as e:
        messages.error(request, f"❌ Error reading dataset: {e}")
        return redirect("/upload/")

    # Step 3: Convert date columns
    for col in ["Date Reported", "Date of Occurrence"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Step 4: Extract year
    if "Date Reported" in df.columns and df["Date Reported"].notna().any():
        df["year"] = df["Date Reported"].dt.year
    elif "Date of Occurrence" in df.columns and df["Date of Occurrence"].notna().any():
        df["year"] = df["Date of Occurrence"].dt.year
    else:
        result = "❌ No valid date column found."
        return render(request, "future.html", {"result": result})

    # Step 5: Clean city/year
    df = df.dropna(subset=["City", "year"])
    cities = sorted(df["City"].dropna().unique().tolist())
    selected_city = request.POST.get("city") if request.method == "POST" else None

    if request.method == "POST":
        # Step 6: Filter by city
        df_filtered = df if selected_city in [None, "All"] else df[df["City"] == selected_city]

        year_counts = df_filtered.groupby("year").size().reset_index(name="count").dropna()
        if len(year_counts) < 2:
            result = f"⚠️ Not enough data to predict for {selected_city or 'All Cities'}."
        else:
            # Step 7: Polynomial Regression (smooth trend)
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.metrics import r2_score
            import numpy as np

            X = year_counts["year"].values.reshape(-1, 1)
            y = year_counts["count"].values

            # Smooth the data slightly
            y = pd.Series(y).rolling(window=2, min_periods=1).mean().values

            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)

            y_pred_train = model.predict(X_poly)
            r2 = r2_score(y, y_pred_train)
            model_accuracy = f"Model Accuracy: {round(r2 * 100, 2)}% (Polynomial Fit)"

            # Step 8: Predict next year
            next_year = int(year_counts["year"].max()) + 1
            next_year_poly = poly.transform([[next_year]])
            pred = model.predict(next_year_poly)[0]
            pred = max(pred, np.mean(y) * 0.5)  # realistic lower bound

            # Step 9: Compute trend and normalized metrics
            trend = y[-1] - y[-2] if len(y) >= 2 else 0
            city_mean = np.mean(y)
            global_mean = df.groupby("City").size().mean()

            normalized_pred = pred / global_mean
            growth_factor = trend / max(global_mean, 1)
            final_score = normalized_pred + (0.25 * np.tanh(growth_factor))

            # Step 10: Risk classification
            if final_score >= 1.3:
                risk_level, risk_color = "High Risk", "#ff4d4d"
            elif final_score >= 0.8:
                risk_level, risk_color = "Medium Risk", "#ffcc00"
            else:
                risk_level, risk_color = "Low Risk", "#00ff88"

            # Step 11: Final result
            predicted_cases = int(round(float(pred), 0))
            result = (
                f"📈 Predicted incidents for {selected_city or 'All Cities'} in {next_year}: "
                f"{predicted_cases} cases ({risk_level}). "
                f"{model_accuracy}"
            )

            # Step 12: Plot visualization
            plt.figure(figsize=(7, 4))
            plt.plot(year_counts["year"], year_counts["count"], marker="o",
                     label="Past Data", linewidth=2)
            plt.scatter([next_year], [predicted_cases], color="red", s=80,
                        label=f"Prediction {next_year}")
            plt.title(f"Crime Trend Prediction - {selected_city or 'All Cities'}", fontsize=12)
            plt.xlabel("Year")
            plt.ylabel("Number of Cases")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            graph_dir = os.path.join(settings.BASE_DIR, "static", "graphs")
            os.makedirs(graph_dir, exist_ok=True)
            graph_path = os.path.join(graph_dir, "future_trend.png")
            plt.savefig(graph_path, dpi=150)
            plt.close()
            line_plot = "/static/graphs/future_trend.png"

    # Step 13: Render
    return render(
        request,
        "future.html",
        {
            "result": result,
            "line_plot": line_plot,
            "cities": ["All"] + cities,
            "selected_city": selected_city,
            "risk_level": risk_level,
            "risk_color": risk_color,
        },
    )

