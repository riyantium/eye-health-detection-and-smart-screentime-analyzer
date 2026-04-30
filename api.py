"""
api.py — Flask backend with user profiles, PDF export, week prediction,
         app categories, heatmap data, two-week comparison,
         forecast accuracy (MAE/RMSE), prediction confidence, data export,
         fatigue detection (OpenCV/MediaPipe),
         and Ocular Health Monitoring (EyeScore, blink kinematics, redness, distance)
Run with: python api.py
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import json, traceback, os, io, csv
from datetime import datetime, timedelta
import tempfile

app = Flask(__name__)
CORS(app)
# CORS 
CORS(app, resources={r"/api/*": {
    "origins": "*",
    "methods": ["GET", "POST", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
    "expose_headers": ["Content-Disposition"],
    "supports_credentials": False,
    "max_age": 3600
}})

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "online"
    })

def handle_options(path):
    response = app.make_response('')
    response.status_code = 204
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Requested-With'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,DELETE,OPTIONS'
    return response

# LAZY IMPORTS 
def try_import_extract():
    try:
        from extract import is_running, get_events, get_daily_totals, get_app_totals
        return is_running, get_events, get_daily_totals, get_app_totals
    except ImportError:
        def is_running(): return False
        def get_events(days=7): return pd.DataFrame()
        def get_daily_totals(df): return pd.DataFrame()
        def get_app_totals(df): return pd.DataFrame()
        return is_running, get_events, get_daily_totals, get_app_totals


def try_import_ml():
    try:
        from ml import (predict_tomorrow, predict_week, detect_anomalies, get_insights,
                        predict_addiction, get_social_media_mins, get_category_breakdown,
                        get_data_sufficiency, get_sleep_impact)
        return (predict_tomorrow, predict_week, detect_anomalies, get_insights,
                predict_addiction, get_social_media_mins, get_category_breakdown,
                get_data_sufficiency, get_sleep_impact)
    except ImportError:
        def predict_tomorrow(df): return 0
        def predict_week(df): return [], {}
        def detect_anomalies(df):
            if df.empty: return df
            df = df.copy()
            df['anomaly'] = False
            df['z_score'] = 0.0
            return df
        def get_insights(daily, apps): return ["Install more days of data for insights."]
        def predict_addiction(avg, apps, social): return "Unknown", 50, ["Track more data for recommendations."], "#6b7280"
        def get_social_media_mins(df): return 0
        def get_category_breakdown(df): return pd.DataFrame(columns=["category", "minutes"])
        def get_data_sufficiency(df): return {}
        def get_sleep_impact(df): return {}
        return (predict_tomorrow, predict_week, detect_anomalies, get_insights,
                predict_addiction, get_social_media_mins, get_category_breakdown,
                get_data_sufficiency, get_sleep_impact)


def try_import_ocular():
    """Returns analyze_video_ocular function; falls back to a safe stub returning
    a well-formed response dict so the frontend never crashes on missing fields."""
    try:
        from ocular import analyze_video_ocular
        return analyze_video_ocular
    except ImportError:
        def analyze_video_ocular(path, **kwargs):
            return {
                "eye_score": 0,
                "risk_level": "Unknown",
                "risk_color": "#6b7280",
                "blink_rate_per_min": None,
                "partial_blink_rate_per_min": None,
                "total_blinks": 0,
                "total_partial_blinks": 0,
                "avg_distance_cm": None,
                "avg_redness": None,
                "squint_count": 0,
                "autocorrect_triggered": False,
                "font_scale_hint": 0,
                "gaze_away_events": 0,
                "gaze_away_total_sec": 0,
                "twenty_twenty_rule_compliant": False,
                "frames_analyzed": 0,
                "duration_sec": 0,
                "recommendations": [
                    "Install the ocular.py module for full eye-health analysis.",
                    "Follow the 20-20-20 rule: every 20 min, look 20 ft away for 20 sec.",
                    "Blink consciously — aim for 15–20 blinks per minute."
                ],
                "disclaimer": "ocular.py module not found. Showing demo/stub values only — not a medical diagnosis.",
                "error": "ocular.py module not found."
            }
        return analyze_video_ocular


def try_import_fatigue():
    """Returns (analyze_video_bytes, result_to_json); falls back to stubs."""
    try:
        from fatigue import analyze_video_bytes, result_to_json
        return analyze_video_bytes, result_to_json
    except ImportError:
        def analyze_video_bytes(path, **kwargs): return None
        def result_to_json(r):
            return {
                "fatigue_score": 0,
                "message": "fatigue.py module not found. Install OpenCV & MediaPipe for fatigue detection.",
                "eye_closure_ratio": None,
                "blink_rate_per_min": None,
                "yawn_ratio": None,
                "frames_analyzed": 0,
                "method": "stub",
                "error": "fatigue.py module not found."
            }
        return analyze_video_bytes, result_to_json


PROFILES_FILE = "profiles.json"

SYSTEM_APPS = {
    "system launcher", "launcher", "android launcher", "com.android.launcher",
    "com.android.systemui", "system ui", "systemui", "android system", "phone",
    "com.android.phone", "android", "unknown", "screen off", "screen on",
    "notification shade", "status bar", "input method", "keyboard", "gboard",
    "com.google.android.inputmethod", "oneplus launcher", "miui home", "huawei home",
    "samsung experience home", "touchwiz home", "com.sec.android.app.launcher"
}


#  PROFILES 

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"profiles": ["Default"], "active": "Default"}


def save_profiles(data):
    try:
        with open(PROFILES_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


@app.route("/api/profiles", methods=["GET"])
def get_profiles():
    return jsonify(load_profiles())


@app.route("/api/profiles", methods=["POST"])
def add_profile():
    try:
        body = request.get_json(silent=True) or {}
        name = body.get("name", "").strip()
        if not name:
            return jsonify({"error": "Name required"}), 400
        data = load_profiles()
        if name not in data["profiles"]:
            data["profiles"].append(name)
            save_profiles(data)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/profiles/active", methods=["POST"])
def set_active_profile():
    try:
        body = request.get_json(silent=True) or {}
        name = body.get("name", "").strip()
        data = load_profiles()
        if name in data["profiles"]:
            data["active"] = name
            save_profiles(data)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/profiles/<name>", methods=["DELETE"])
def delete_profile(name):
    try:
        data = load_profiles()
        if name in data["profiles"] and name != "Default":
            data["profiles"].remove(name)
            if data["active"] == name:
                data["active"] = "Default"
            save_profiles(data)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# FILE PARSING 

def extract_hour(timestamp):
    try:
        ts = str(timestamp)
        if len(ts) >= 13:
            return int(ts[11:13])
    except Exception:
        pass
    return -1


def parse_phone_file(file):
    filename = file.filename.lower() if file.filename else ""
    rows = []
    try:
        if filename.endswith('.json'):
            raw = file.read()
            data = json.loads(raw)
            buckets = data.get("buckets", data) if isinstance(data, dict) else {}
            if isinstance(buckets, dict):
                for bucket_id, bucket in buckets.items():
                    events = bucket.get("events", []) if isinstance(bucket, dict) else []
                    for event in events:
                        app_name = (event.get("data", {}).get("app") or
                                    event.get("data", {}).get("title") or "Unknown")
                        duration = event.get("duration", 0) or 0
                        timestamp = event.get("timestamp", "") or ""
                        date = timestamp[:10] if timestamp else "phone"
                        rows.append({"date": date, "app": str(app_name),
                                     "minutes": round(float(duration) / 60, 2),
                                     "hour": extract_hour(timestamp)})
            elif isinstance(data, list):
                for event in data:
                    app_name = (event.get("data", {}).get("app") if isinstance(event.get("data"), dict) else None) \
                               or event.get("app") or "Unknown"
                    duration = event.get("duration", 0) or 0
                    timestamp = event.get("timestamp", "") or ""
                    date = timestamp[:10] if timestamp else "phone"
                    rows.append({"date": date, "app": str(app_name),
                                 "minutes": round(float(duration) / 60, 2),
                                 "hour": extract_hour(timestamp)})

        elif filename.endswith('.csv'):
            import re
            file.seek(0)
            df = pd.read_csv(file)
            df.columns = [c.lower().strip() for c in df.columns]
            app_col = next((c for c in df.columns if 'app' in c or 'name' in c), None)
            time_col = next((c for c in df.columns if any(k in c for k in
                            ['duration', 'time', 'minutes', 'usage'])), None)
            date_col = next((c for c in df.columns if 'date' in c or 'day' in c), None)
            ts_col = next((c for c in df.columns if 'timestamp' in c or c == 'time'), None)
            hour_col = next((c for c in df.columns if c == 'hour'), None)

            if app_col and time_col:
                for _, row in df.iterrows():
                    raw = str(row[time_col]) if pd.notna(row[time_col]) else "0"
                    hm = re.match(r'(\d+)h\s*(\d+)m', raw)
                    ho = re.match(r'(\d+)h', raw)
                    mo = re.match(r'(\d+)m', raw)
                    if hm:
                        mins = int(hm.group(1)) * 60 + int(hm.group(2))
                    elif ho:
                        mins = int(ho.group(1)) * 60
                    elif mo:
                        mins = int(mo.group(1))
                    else:
                        try:
                            mins = float(raw)
                            if mins > 1000:
                                mins /= 60
                        except Exception:
                            mins = 0

                    date = str(row[date_col]) if date_col and pd.notna(row.get(date_col)) else "phone"
                    if hour_col and pd.notna(row.get(hour_col)):
                        try:
                            hour = int(row[hour_col])
                        except Exception:
                            hour = -1
                    elif ts_col and pd.notna(row.get(ts_col)):
                        hour = extract_hour(str(row[ts_col]))
                    else:
                        hour = -1
                    rows.append({"date": date, "app": str(row[app_col]),
                                 "minutes": round(float(mins), 2), "hour": hour})
    except Exception as e:
        print(f"[parse_phone_file] Error: {e}")
        traceback.print_exc()

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "app", "minutes", "hour"])


def clean_df(df):
    if df.empty:
        return df
    if "hour" not in df.columns:
        df = df.copy()
        df["hour"] = -1
    df = df[df["minutes"] > 0].copy()
    df = df[~df["app"].str.lower().isin(SYSTEM_APPS)]
    df = df[~df["app"].str.lower().str.contains(
        r"launcher|systemui|android\.system|com\.android", na=False, regex=True)]
    return df


def get_heatmap_data(df):
    if df.empty:
        return []
    daily = df.groupby("date")["minutes"].sum().reset_index()
    try:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily.dropna(subset=["date"])
    except Exception:
        return []
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=90))
    daily = daily[daily["date"] >= cutoff]
    return [{"date": str(r["date"].date()), "minutes": round(float(r["minutes"]))}
            for _, r in daily.iterrows()]


def build_response(df, days):
    is_running, get_events, get_daily_totals, get_app_totals = try_import_extract()
    (predict_tomorrow, predict_week, detect_anomalies, get_insights,
     predict_addiction, get_social_media_mins, get_category_breakdown,
     get_data_sufficiency, get_sleep_impact) = try_import_ml()

    daily_df = get_daily_totals(df)
    app_df = get_app_totals(df)
    anomaly_df = detect_anomalies(daily_df)
    week_pred, accuracy = predict_week(daily_df)

    if not isinstance(accuracy, dict):
        accuracy = {}

    prediction = week_pred[0]["minutes"] if week_pred else 0
    week_pred_rf = accuracy.get("predictions_rf") or []
    week_pred_ens = accuracy.get("predictions_ensemble") or []
    prediction_rf = week_pred_rf[0]["minutes"] if week_pred_rf else 0
    prediction_ens = week_pred_ens[0]["minutes"] if week_pred_ens else prediction
    insights = get_insights(daily_df, app_df)

    if not daily_df.empty and "minutes" in daily_df.columns:
        daily_df = daily_df.copy()
        daily_df["rolling3"] = daily_df["minutes"].rolling(3, min_periods=1).mean().round(1)
    elif not daily_df.empty:
        daily_df = daily_df.copy()
        daily_df["rolling3"] = 0.0

    top5 = app_df.head(5)["app"].tolist() if not app_df.empty else []
    trends = {}
    for a in top5:
        subset = df[df["app"] == a]
        if not subset.empty:
            trends[a] = subset.groupby("date")["minutes"].sum().to_dict()

    avg_daily = float(daily_df["minutes"].mean()) if not daily_df.empty else 0
    social_mins = get_social_media_mins(app_df)
    addiction_level, confidence, tips, addiction_color = predict_addiction(
        avg_daily, len(app_df), social_mins)

    category_df = get_category_breakdown(app_df)
    heatmap = get_heatmap_data(df)
    sleep_impact = get_sleep_impact(df)

    total_events = len(df)
    unique_apps = int(df["app"].nunique()) if not df.empty else 0
    date_range = len(daily_df)

    anomaly_records = []
    if not anomaly_df.empty:
        for _, row in anomaly_df.iterrows():
            anomaly_records.append({
                "date": str(row.get("date", "")),
                "minutes": round(float(row.get("minutes", 0)), 1),
                "anomaly": bool(row.get("anomaly", False)),
                "z_score": float(row.get("z_score", 0))
            })

    rolling_records = []
    if not daily_df.empty and "rolling3" in daily_df.columns:
        rolling_records = daily_df[["date", "rolling3"]].fillna(0).to_dict(orient="records")

    apps_records = []
    if not app_df.empty:
        app_df_copy = app_df.head(20).copy()
        app_df_copy["minutes"] = app_df_copy["minutes"].round(1)
        apps_records = app_df_copy.to_dict(orient="records")

    cats_records = []
    if not category_df.empty:
        cats_records = category_df.to_dict(orient="records")

    return {
        "daily": anomaly_records,
        "apps": apps_records,
        "rolling": rolling_records,
        "trends": trends,
        "week_prediction": week_pred,
        "week_prediction_rf": week_pred_rf,
        "week_prediction_ensemble": week_pred_ens,
        "prediction": float(prediction),
        "prediction_rf": float(prediction_rf),
        "prediction_ensemble": float(prediction_ens),
        "prediction_note": "Linear Regression + Random Forest (7-day forecast)",
        "prediction_confidence": accuracy.get("confidence", 60),
        "forecast_accuracy": accuracy,
        "insights": insights,
        "total_minutes": round(float(df["minutes"].sum())) if not df.empty else 0,
        "avg_minutes": round(avg_daily),
        "top_app": app_df.iloc[0]["app"] if not app_df.empty else "N/A",
        "days": days,
        "addiction_level": addiction_level,
        "addiction_confidence": confidence,
        "addiction_tips": tips,
        "addiction_color": addiction_color,
        "categories": cats_records,
        "heatmap": heatmap,
        "sleep_impact": sleep_impact,
        "data_stats": {
            "total_events": total_events,
            "unique_apps": unique_apps,
            "days_tracked": date_range,
            "local_only": True
        }
    }


# ROUTES 

@app.route("/api/status")
def status():
    try:
        is_running, _, _, _ = try_import_extract()
        return jsonify({"running": is_running(), "status": "ok"})
    except Exception as e:
        return jsonify({"running": False, "status": "ok", "note": str(e)})


@app.route("/api/data", methods=["POST"])
def data():
    try:
        is_running, get_events, get_daily_totals, get_app_totals = try_import_extract()
        days = int(request.form.get("days", 7))
        include_laptop = request.form.get("include_laptop", "true").lower() == "true"
        phone_file = request.files.get("phone_file")
        frames = []

        if include_laptop:
            if is_running():
                laptop_df = get_events(days=days)
                if not laptop_df.empty:
                    laptop_df = laptop_df.copy()
                    laptop_df["source"] = "laptop"
                    frames.append(laptop_df)

        if phone_file and phone_file.filename:
            phone_df = parse_phone_file(phone_file)
            if not phone_df.empty:
                phone_df = phone_df.copy()
                phone_df["source"] = "phone"
                cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                phone_df = phone_df[phone_df["date"] >= cutoff]
                if not phone_df.empty:
                    frames.append(phone_df)

        if not frames:
            if include_laptop and not is_running():
                return jsonify({"error": "ActivityWatch is not running. Please start it or upload phone data."}), 400
            return jsonify({"error": "No data found. Please check your data sources."}), 404

        combined = clean_df(pd.concat(frames, ignore_index=True))
        if combined.empty:
            return jsonify({"error": "No usable data after filtering system apps."}), 404

        return jsonify(build_response(combined, days))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    try:
        is_running, get_events, _, _ = try_import_extract()
        include_laptop = request.form.get("include_laptop", "true").lower() == "true"
        phone_file = request.files.get("phone_file")
        frames = []

        if include_laptop and is_running():
            laptop_df = get_events(days=30)
            if not laptop_df.empty:
                frames.append(laptop_df)

        if phone_file and phone_file.filename:
            phone_file.seek(0)
            phone_df = parse_phone_file(phone_file)
            if not phone_df.empty:
                frames.append(phone_df)

        if not frames:
            return jsonify({"error": "No data found"}), 404

        df = clean_df(pd.concat(frames, ignore_index=True))
        if df.empty:
            return jsonify({"error": "No usable data after filtering."}), 404

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        now = datetime.now()
        week1_start = pd.Timestamp(now - timedelta(days=14))
        week1_end = pd.Timestamp(now - timedelta(days=7))
        week2_start = pd.Timestamp(now - timedelta(days=7))
        week2_end = pd.Timestamp(now)

        w1 = df[(df["date"] >= week1_start) & (df["date"] < week1_end)]
        w2 = df[(df["date"] >= week2_start) & (df["date"] <= week2_end)]

        def week_summary(wdf):
            if wdf.empty:
                return {"total": 0, "avg": 0, "top_app": "N/A", "apps": [], "daily": []}
            daily = wdf.groupby(wdf["date"].dt.strftime("%Y-%m-%d"))["minutes"].sum()
            apps = wdf.groupby("app")["minutes"].sum().sort_values(ascending=False).head(5)
            return {
                "total": round(float(wdf["minutes"].sum())),
                "avg": round(float(wdf["minutes"].sum()) / 7),
                "top_app": apps.index[0] if not apps.empty else "N/A",
                "apps": [{"app": a, "minutes": round(float(m))} for a, m in apps.items()],
                "daily": [{"date": d, "minutes": round(float(m))} for d, m in daily.items()]
            }

        return jsonify({
            "week1": {**week_summary(w1),
                      "label": (now - timedelta(days=14)).strftime("%b %d") + " – " + (now - timedelta(days=7)).strftime("%b %d")},
            "week2": {**week_summary(w2),
                      "label": (now - timedelta(days=7)).strftime("%b %d") + " – " + now.strftime("%b %d")}
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# DATA EXPORT 

@app.route("/api/export_csv", methods=["POST"])
def export_csv():
    try:
        is_running, get_events, _, _ = try_import_extract()
        include_laptop = request.form.get("include_laptop", "true").lower() == "true"
        phone_file = request.files.get("phone_file")
        days = int(request.form.get("days", 30))
        frames = []

        if include_laptop and is_running():
            laptop_df = get_events(days=days)
            if not laptop_df.empty:
                laptop_df = laptop_df.copy()
                laptop_df["source"] = "laptop"
                frames.append(laptop_df)

        if phone_file and phone_file.filename:
            phone_file.seek(0)
            phone_df = parse_phone_file(phone_file)
            if not phone_df.empty:
                phone_df = phone_df.copy()
                phone_df["source"] = "phone"
                frames.append(phone_df)

        if not frames:
            return jsonify({"error": "No data found"}), 404

        combined = clean_df(pd.concat(frames, ignore_index=True))
        output = io.StringIO()
        combined.to_csv(output, index=False)
        output.seek(0)
        buf = io.BytesIO(output.getvalue().encode("utf-8"))
        buf.seek(0)
        return send_file(buf, mimetype='text/csv', as_attachment=True,
                         download_name='my_screen_time_data_' + datetime.now().strftime('%Y%m%d') + '.csv')
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# PDF EXPORT 

@app.route("/api/export_pdf", methods=["POST"])
def export_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 400

    try:
        payload = request.get_json(silent=True) or {}
        report_data = payload.get("data", {})
        ocular = report_data.get("ocular", {}) or {}
        fatigue = report_data.get("fatigue", {}) or {}
        user = payload.get("user", "Default")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=2 * cm, leftMargin=2 * cm,
                                topMargin=2 * cm, bottomMargin=2 * cm)

        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle('title', fontSize=22, fontName='Helvetica-Bold',
                                     textColor=colors.HexColor('#1a1a2e'),
                                     spaceAfter=4, alignment=TA_CENTER)
        sub_style = ParagraphStyle('sub', fontSize=11, fontName='Helvetica',
                                   textColor=colors.HexColor('#6b7280'),
                                   spaceAfter=2, alignment=TA_CENTER)
        h2_style = ParagraphStyle('h2', fontSize=14, fontName='Helvetica-Bold',
                                  textColor=colors.HexColor('#1a1a2e'),
                                  spaceBefore=16, spaceAfter=8)
        body_style = ParagraphStyle('body', fontSize=10, fontName='Helvetica',
                                    textColor=colors.HexColor('#374151'),
                                    spaceAfter=4, leading=16)

        story.append(Paragraph("Screen Time Report", title_style))
        story.append(Paragraph(
            "User: " + str(user) + "  |  Generated: " + datetime.now().strftime('%B %d, %Y'), sub_style))
        story.append(Paragraph(
            "All data is processed locally. Your data never leaves your device.", sub_style))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor('#e5e7eb'), spaceAfter=16))

        story.append(Paragraph("Summary", h2_style))
        total = report_data.get("total_minutes", 0) or 0
        avg = report_data.get("avg_minutes", 0) or 0
        top_app = report_data.get("top_app", "N/A") or "N/A"
        days = report_data.get("days", 7)
        addiction = report_data.get("addiction_level", "N/A") or "N/A"
        acc = report_data.get("forecast_accuracy", {}) or {}
        mae = acc.get("mae")
        conf = report_data.get("prediction_confidence", "N/A")
        suff = acc.get("sufficiency", {}) or {}

        total = int(total)
        avg = int(avg)

        metrics_data = [
            ["Metric", "Value"],
            ["Period analyzed", "Last " + str(days) + " days"],
            ["Total screen time", str(total // 60) + "h " + str(total % 60) + "m"],
            ["Daily average", str(avg // 60) + "h " + str(avg % 60) + "m"],
            ["Most used app", str(top_app)],
            ["Addiction level", str(addiction)],
            ["Forecast confidence", str(conf) + "%"],
            ["Forecast MAE", (str(mae) + " min" if mae is not None else "N/A (need 7+ days)")],
            ["Model quality", str(suff.get("label", "N/A")) + " (" + str(suff.get("days", "?")) + " days)"],
        ]
        t = Table(metrics_data, colWidths=[8 * cm, 8 * cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor('#f9fafb'), colors.HexColor('#ffffff')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]))
        story.append(t)
        story.append(Spacer(1, 16))

        # OCULAR HEALTH 
        story.append(Paragraph("Ocular Health Analysis", h2_style))

        ocular_data = [
            ["Metric", "Value"],
            ["Blink Rate", f"{ocular.get('blink_rate_per_min', 'N/A')} bpm"],
            ["Partial Blink", f"{ocular.get('partial_blink_rate_per_min', 'N/A')} /min"],
            ["Redness", f"{round((ocular.get('avg_redness', 0))*100)} %"],
            ["Distance", f"{ocular.get('avg_distance_cm', 'N/A')} cm"],
            ["Squints", f"{ocular.get('squint_count', 'N/A')} frames"],
            ["Eye Score", f"{ocular.get('eye_score', 'N/A')} /10"],
        ]

        t_ocular = Table(ocular_data, colWidths=[8 * cm, 8 * cm])
        t_ocular.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0ea5e9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))

        story.append(t_ocular)
        story.append(Spacer(1, 16))

        # ─── FATIGUE ─────────────────────────────
        story.append(Paragraph("Fatigue Analysis", h2_style))

        fatigue_data = [
            ["Metric", "Value"],
            ["Fatigue Score", f"{fatigue.get('fatigue_score', 'N/A')} /100"],
            ["Blink Rate", f"{fatigue.get('blink_rate_per_min', 'N/A')} bpm"],
            ["Eye Closure Ratio", f"{fatigue.get('eye_closure_ratio', 'N/A')}"],
            ["Yawn Ratio", f"{fatigue.get('yawn_ratio', 'N/A')}"],
        ]

        t_fatigue = Table(fatigue_data, colWidths=[8 * cm, 8 * cm])
        t_fatigue.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f59e0b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))

        story.append(t_fatigue)
        story.append(Spacer(1, 16))

        apps = report_data.get("apps", []) or []
        if apps:
            story.append(Paragraph("Top Apps", h2_style))
            app_data = [["Rank", "App", "Time", "% of Total"]]
            total_mins = sum(a.get("minutes", 0) for a in apps) or 1
            for i, a in enumerate(apps[:10], 1):
                m = float(a.get("minutes", 0))
                pct = round(m / total_mins * 100, 1)
                app_data.append([str(i), str(a.get("app", "")),
                                  str(int(m // 60)) + "h " + str(int(m % 60)) + "m",
                                  str(pct) + "%"])
            t2 = Table(app_data, colWidths=[2 * cm, 8 * cm, 4 * cm, 3 * cm])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f8ef7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.HexColor('#f0f9ff'), colors.HexColor('#ffffff')]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
                ('PADDING', (0, 0), (-1, -1), 7),
                ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(t2)
            story.append(Spacer(1, 16))

        insights = report_data.get("insights", []) or []
        if insights:
            story.append(Paragraph("Insights", h2_style))
            for ins in insights:
                story.append(Paragraph("• " + str(ins), body_style))
            story.append(Spacer(1, 8))

        tips = report_data.get("addiction_tips", []) or []
        if tips:
            story.append(Paragraph("Recommendations", h2_style))
            for tip in tips:
                story.append(Paragraph("• " + str(tip), body_style))

        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor('#e5e7eb'), spaceAfter=8))
        story.append(Paragraph(
            "Privacy: All data in this report was collected and processed entirely on your local device. "
            "No data was transmitted to any external server. You own your data.",
            ParagraphStyle('privacy', fontSize=9, fontName='Helvetica',
                           textColor=colors.HexColor('#9ca3af'), alignment=TA_CENTER)
        ))

        doc.build(story)
        buffer.seek(0)

        return send_file(buffer, mimetype='application/pdf',
                         as_attachment=True,
                         download_name="screen_time_report_" + str(user) + "_" + datetime.now().strftime('%Y%m%d') + ".pdf")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# FATIGUE 

@app.route("/api/fatigue", methods=["POST"])
def fatigue():
    try:
        analyze_video_bytes, result_to_json = try_import_fatigue()
        up = request.files.get("video")
        if not up:
            return jsonify({"error": "Missing 'video' file upload"}), 400

        suffix = os.path.splitext(up.filename or "")[1].lower()
        if suffix not in {".webm", ".mp4", ".mov", ".avi", ".mkv"}:
            suffix = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            tmp_path = f.name
            up.save(tmp_path)

        try:
            result = analyze_video_bytes(tmp_path, max_seconds=6.0)
            return jsonify(result_to_json(result))
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# OCULAR HEALTH 
@app.route("/api/ocular", methods=["POST"])
def ocular():
    try:
        analyze_video_ocular = try_import_ocular()
        up = request.files.get("video")
        if not up:
            return jsonify({"error": "Missing 'video' file upload"}), 400

        try:
            screen_time_hours = float(request.form.get("screen_time_hours", 0.0) or 0.0)
        except (ValueError, TypeError):
            screen_time_hours = 0.0

        try:
            focal_px = float(request.form.get("focal_px", 600.0) or 600.0)
        except (ValueError, TypeError):
            focal_px = 600.0

        suffix = os.path.splitext(up.filename or "")[1].lower()
        if suffix not in {".webm", ".mp4", ".mov", ".avi", ".mkv"}:
            suffix = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            tmp_path = f.name
            up.save(tmp_path)

        try:
            result = analyze_video_ocular(
                tmp_path,
                max_seconds=10.0,
                focal_px=focal_px,
                screen_time_hours=screen_time_hours,
            )
            if not isinstance(result, dict):
                result = {"error": "Unexpected result format from ocular module."}
            return jsonify(result)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# HEALTH CHECK 

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


# ENTRY POINT

if __name__ == "__main__":
    print("=" * 50)
    print("  Screen Time API — http://localhost:5000")
    print("  CORS: enabled for all origins")
    print("=" * 50)
    app.run(
        debug=True,
        port=5000,
        host="0.0.0.0",
        use_reloader=True,
        threaded=True
    )
