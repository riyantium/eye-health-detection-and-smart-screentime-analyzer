"""
ml.py — Machine Learning for Screen Time Analysis
Includes: addiction prediction, 7-day forecast, anomaly detection,
          forecast accuracy (MAE/RMSE), prediction confidence, data sufficiency
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

KAGGLE_PATH = "kaggle_ref_addiction.csv"

APP_CATEGORIES = {
    "Social": ["instagram","facebook","twitter","tiktok","snapchat","whatsapp",
               "telegram","discord","threads","linkedin","pinterest","tumblr","x","wechat","viber"],
    "Entertainment": ["youtube","netflix","spotify","prime video","hotstar","zee5","mx player",
                      "jio cinema","apple tv","twitch","soundcloud","gaana","wynk"],
    "Gaming": ["clash of clans","pubg","freefire","genshin","roblox","minecraft","fortnite",
               "steam","epic games","mobile legends","cod","among us","candy crush","ludo"],
    "Productivity": ["gmail","outlook","docs","sheets","excel","word","powerpoint","notion",
                     "slack","teams","zoom","meet","calendar","drive","dropbox","trello","asana"],
    "Browser": ["chrome","firefox","safari","edge","opera","brave","duckduckgo"],
    "Education": ["duolingo","khan academy","coursera","udemy","byjus","unacademy","vedantu","photomath"],
    "Health": ["fitbit","strava","nike run","samsung health","google fit","headspace","calm","myfitnesspal"],
    "Shopping": ["amazon","flipkart","meesho","myntra","ajio","nykaa","swiggy","zomato","blinkit"],
    "Finance": ["gpay","phonepe","paytm","cred","zerodha","groww","bhim"],
}

ADDICTION_TIPS = {
    "Low":      ["Great job! Your screen time is healthy.",
                 "Keep taking regular breaks every hour.",
                 "Stay active — physical activity maintains healthy screen habits."],
    "Mild":     ["Your usage is slightly above ideal. Try setting daily limits.",
                 "Use Digital Wellbeing to set app timers.",
                 "Try the 20-20-20 rule: every 20 mins, look 20 feet away for 20 seconds."],
    "Moderate": ["Consider reducing screen time by 30 mins/day.",
                 "Turn off non-essential notifications.",
                 "Set phone-free hours — especially 1 hour before bed.",
                 "Replace 30 mins of screen time with a walk or hobby."],
    "High":     ["Your screen time is high and may affect health and productivity.",
                 "Use app blockers like StayFree or Digital Wellbeing.",
                 "Delete or move addictive apps off your home screen.",
                 "Tell a friend about your goal to reduce screen time.",
                 "Replace evening screen time with reading or journaling."],
    "Extreme":  ["Your screen usage is extreme. Take action now.",
                 "Try a digital detox — start with 1 screen-free day per week.",
                 "Remove social media apps from your phone.",
                 "Set strict bedtime rules — no screens after 9 PM.",
                 "Seek professional guidance if affecting sleep or relationships."]
}

ADDICTION_COLORS = {
    "Low": "#34d399", "Mild": "#fbbf24",
    "Moderate": "#fb923c", "High": "#f87171", "Extreme": "#ef4444"
}

# When user enables eye/health mode, classify risk as if daily usage were higher
# (stricter guidance for glasses, dry eyes, myopia, doctor advice, etc.)
EYE_HEALTH_USAGE_MULTIPLIER = 1.35

EYE_HEALTH_EXTRA_TIPS = [
    "You enabled eye/health mode: aim for shorter daily screen time than typical guidance.",
    "Follow the 20-20-20 rule more strictly: every 20 minutes, look 20 feet away for 20 seconds.",
    "Keep brightness comfortable; avoid long sessions in dark rooms.",
    "If you wear glasses or have a prescription, keep regular check-ups and follow your optometrist’s screen-time advice.",
]

SOCIAL_APPS = {"instagram","facebook","twitter","tiktok","snapchat","whatsapp",
               "telegram","discord","threads","linkedin","pinterest","x","youtube","reddit"}


def categorize_app(app_name):
    name = app_name.lower().strip()
    for category, apps in APP_CATEGORIES.items():
        if any(a in name for a in apps):
            return category
    return "Other"


def get_category_breakdown(app_df):
    app_df = app_df.copy()
    app_df["category"] = app_df["app"].apply(categorize_app)
    return app_df.groupby("category")["minutes"].sum().reset_index().sort_values("minutes", ascending=False)


# ─── TRAIN MODEL ──────────────────────────────────────────────────────────────

def train_model():
    if not os.path.exists(KAGGLE_PATH):
        return None, None, []
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        df = pd.read_csv(KAGGLE_PATH)
        available = df.columns.tolist()
        desired = ['usage_per_day_minutes', 'notifications_received', 'times_opened',
                   'social_media_or_gaming_hours', 'sleep_hours', 'work_hours',
                   'academic_professional_performance', 'addiction_score']
        features = [f for f in desired if f in available]
        X = df[features].fillna(0)
        y = df['addiction_level']
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X, y_enc)
        return model, le, features
    except Exception as e:
        print("Model training failed:", e)
        return None, None, []


_model, _le, _features = train_model()


# ─── RULE-BASED FALLBACK ──────────────────────────────────────────────────────

def rule_based_addiction(usage_per_day_mins, social_media_mins, eye_health_sensitive=False):
    u = usage_per_day_mins * (EYE_HEALTH_USAGE_MULTIPLIER if eye_health_sensitive else 1.0)
    social_ratio = social_media_mins / max(usage_per_day_mins, 1)
    if u < 120:
        level = "Low"
    elif u < 240:
        level = "Mild"
    elif u < 360:
        level = "Moderate"
    elif u < 480:
        level = "High"
    else:
        level = "Extreme"
    levels = ["Low", "Mild", "Moderate", "High", "Extreme"]
    if social_ratio > 0.5 and levels.index(level) < 4:
        level = levels[levels.index(level) + 1]
    return level, 70.0


def _merge_addiction_tips(level, eye_health_sensitive):
    base = list(ADDICTION_TIPS.get(level, []))
    if eye_health_sensitive:
        return EYE_HEALTH_EXTRA_TIPS + base
    return base


# ─── ADDICTION PREDICTION ─────────────────────────────────────────────────────

def predict_addiction(usage_per_day_mins, num_apps, social_media_mins, eye_health_sensitive=False):
    u_for_model = usage_per_day_mins * (EYE_HEALTH_USAGE_MULTIPLIER if eye_health_sensitive else 1.0)
    if _model is None:
        level, confidence = rule_based_addiction(usage_per_day_mins, social_media_mins, eye_health_sensitive)
        return level, confidence, _merge_addiction_tips(level, eye_health_sensitive), ADDICTION_COLORS.get(level, "#888")
    try:
        notifications = int(u_for_model * 0.8)
        times_opened = num_apps * 3
        social_hours = social_media_mins / 60
        sleep_hours = max(4, 8 - u_for_model / 120)
        work_hours = 8.0
        academic_perf = max(50, 100 - (u_for_model / 10))
        addiction_score = min(10, u_for_model / 60)
        all_vals = {
            'usage_per_day_minutes': u_for_model,
            'notifications_received': notifications,
            'times_opened': times_opened,
            'social_media_or_gaming_hours': social_hours,
            'sleep_hours': sleep_hours,
            'work_hours': work_hours,
            'academic_professional_performance': academic_perf,
            'addiction_score': addiction_score
        }
        X = np.array([[all_vals[f] for f in _features]])
        pred = _model.predict(X)[0]
        proba = _model.predict_proba(X)[0]
        confidence = round(float(np.max(proba)) * 100, 1)
        level = _le.inverse_transform([pred])[0]
        return level, confidence, _merge_addiction_tips(level, eye_health_sensitive), ADDICTION_COLORS.get(level, "#888")
    except Exception as e:
        print("Addiction prediction error:", e)
        level, confidence = rule_based_addiction(usage_per_day_mins, social_media_mins, eye_health_sensitive)
        return level, confidence, _merge_addiction_tips(level, eye_health_sensitive), ADDICTION_COLORS.get(level, "#888")


def suggested_daily_cap_minutes(eye_health_sensitive=False):
    """Soft target for UI — not medical advice."""
    return 150 if eye_health_sensitive else 240


# ─── DATE PARSING HELPER ──────────────────────────────────────────────────────

def safe_parse_dates(daily_df):
    """
    Safely parse dates in any format.
    Handles: YYYY-MM-DD, DD-MM-YYYY, MM/DD/YYYY, YYYY/MM/DD,
             'phone', 'unknown', and any other weird formats.
    Returns a clean DataFrame with only valid real dates.
    """
    df = daily_df.copy()

    # Remove rows where date is clearly not a date
    bad_dates = {"phone", "unknown", "nan", "none", "", "null"}
    df = df[~df["date"].astype(str).str.lower().isin(bad_dates)]

    if df.empty:
        return df

    # Try to parse dates flexibly
    try:
        df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
    except Exception:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows where date could not be parsed
    df = df.dropna(subset=["date"])

    if df.empty:
        return df

    # Convert back to string format YYYY-MM-DD for consistency
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Re-aggregate by date in case same date appeared in multiple formats
    df = df.groupby("date")["minutes"].sum().reset_index().sort_values("date")

    return df


# ─── FORECAST ─────────────────────────────────────────────────────────────────

def predict_week(daily_df):
    """
    Predict next 7 days.
    Returns (predictions list, accuracy dict).
    Falls back to rolling average if not enough clean data.
    """
    # Always compute average from raw data first as safe fallback
    raw_avg = daily_df["minutes"].mean() if not daily_df.empty else 297
    raw_avg = max(1, round(raw_avg))  # never predict 0

    def fallback(reason=""):
        if reason:
            print("Forecast fallback:", reason)
        preds = [{"date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                  "minutes": raw_avg} for i in range(7)]
        return preds, {
            "mae": None, "rmse": None, "confidence": 60,
            "sufficiency": get_data_sufficiency(len(daily_df))
        }

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error
        import math

        # Clean and parse dates properly
        df = safe_parse_dates(daily_df)

        if df.empty:
            return fallback("no valid dates after parsing")

        # Need at least 3 data points to build lag features
        if len(df) < 3:
            return fallback("less than 3 days of data")

        df = df.sort_values("date").reset_index(drop=True)

        # Build features
        df["date_dt"] = pd.to_datetime(df["date"])
        df["dayofweek"] = df["date_dt"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["lag1"] = df["minutes"].shift(1)
        df["lag2"] = df["minutes"].shift(2)
        df["rolling3"] = df["minutes"].rolling(3).mean()

        df_model = df.dropna(subset=["lag1", "lag2", "rolling3"]).copy()

        if len(df_model) < 2:
            return fallback("not enough rows after lag features")

        features = ["dayofweek", "is_weekend", "lag1", "lag2", "rolling3"]
        X = df_model[features].values
        y = df_model["minutes"].values

        # Sanity check — if all y values are 0 something is wrong
        if y.max() == 0:
            return fallback("all target values are 0")

        # Linear Regression (scaled)
        scaler = StandardScaler()
        model = LinearRegression()

        # Random Forest (no scaling needed)
        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )

        # Walk-forward validation for accuracy metrics
        mae_val, rmse_val = None, None
        mae_rf, rmse_rf = None, None
        if len(df_model) >= 7:
            split = max(3, len(df_model) - 4)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]
            sc_val = StandardScaler()
            m_val = LinearRegression()
            m_val.fit(sc_val.fit_transform(X_tr), y_tr)
            y_pred_val = np.maximum(m_val.predict(sc_val.transform(X_te)), 0)
            mae_val = round(float(mean_absolute_error(y_te, y_pred_val)), 1)
            rmse_val = round(float(math.sqrt(np.mean((y_te - y_pred_val) ** 2))), 1)

            # RF validation
            rf_val = RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1,
            )
            rf_val.fit(X_tr, y_tr)
            y_pred_rf = np.maximum(rf_val.predict(X_te), 0)
            mae_rf = round(float(mean_absolute_error(y_te, y_pred_rf)), 1)
            rmse_rf = round(float(math.sqrt(np.mean((y_te - y_pred_rf) ** 2))), 1)

        # Train on full data
        model.fit(scaler.fit_transform(X), y)
        rf.fit(X, y)

        # Predict next 7 days (both models + simple average ensemble)
        predictions = []
        predictions_rf = []
        predictions_ens = []
        last_mins = float(df["minutes"].values[-1])
        second_last = float(df["minutes"].values[-2]) if len(df) >= 2 else last_mins
        rolling = float(df["minutes"].tail(3).mean())

        for i in range(7):
            d = datetime.now() + timedelta(days=i + 1)
            feat = np.array([[
                d.weekday(),
                1 if d.weekday() >= 5 else 0,
                last_mins,
                second_last,
                rolling
            ]])
            raw_pred = float(model.predict(scaler.transform(feat))[0])
            raw_pred_rf = float(rf.predict(feat)[0])

            # Key fix: clamp prediction between 10% and 300% of average
            # This prevents 0m or unrealistically large predictions
            pred = round(max(raw_avg * 0.1, min(raw_avg * 3.0, raw_pred)))
            pred_rf = round(max(raw_avg * 0.1, min(raw_avg * 3.0, raw_pred_rf)))
            pred_ens = round((pred + pred_rf) / 2.0)

            predictions.append({"date": d.strftime("%Y-%m-%d"), "minutes": pred})
            predictions_rf.append({"date": d.strftime("%Y-%m-%d"), "minutes": pred_rf})
            predictions_ens.append({"date": d.strftime("%Y-%m-%d"), "minutes": pred_ens})
            second_last = last_mins
            last_mins = float(pred_ens)
            rolling = (rolling * 2 + pred_ens) / 3

        # Confidence score
        avg = df["minutes"].mean()
        if mae_val is not None and avg > 0:
            error_ratio = mae_val / avg
            confidence = round(max(30, min(95, (1 - error_ratio) * 100)), 1)
        else:
            confidence = 65

        accuracy = {
            "mae": mae_val,
            "rmse": rmse_val,
            "mae_rf": mae_rf,
            "rmse_rf": rmse_rf,
            "confidence": confidence,
            "sufficiency": get_data_sufficiency(len(daily_df))
        }

        # Keep backward-compatibility: return LR predictions as the primary list.
        # API layer can optionally expose predictions_rf / ensemble.
        accuracy["predictions_rf"] = predictions_rf
        accuracy["predictions_ensemble"] = predictions_ens
        return predictions, accuracy

    except Exception as e:
        print("Forecast error:", e)
        return fallback(str(e))


def get_data_sufficiency(num_days):
    if num_days >= 30:
        return {"label": "Excellent", "color": "#34d399", "days": num_days,
                "message": str(num_days) + " days of data — model is highly accurate"}
    elif num_days >= 14:
        return {"label": "Good", "color": "#4f8ef7", "days": num_days,
                "message": str(num_days) + " days of data — model is reliable"}
    elif num_days >= 7:
        return {"label": "Fair", "color": "#fbbf24", "days": num_days,
                "message": str(num_days) + " days of data — accuracy improves with more data"}
    else:
        return {"label": "Low", "color": "#f87171", "days": num_days,
                "message": str(num_days) + " days of data — collect more data for better predictions"}


def predict_tomorrow(daily_df):
    week, accuracy = predict_week(daily_df)
    return week[0]["minutes"], "Linear Regression", accuracy


# ─── ANOMALY DETECTION ────────────────────────────────────────────────────────

def detect_anomalies(daily_df):
    df = daily_df.copy()
    if len(df) < 3:
        return df.assign(anomaly=False, z_score=0.0)
    mean, std = df["minutes"].mean(), df["minutes"].std()
    if std == 0:
        return df.assign(anomaly=False, z_score=0.0)
    df["z_score"] = ((df["minutes"] - mean) / std).round(2)
    df["anomaly"] = df["z_score"].abs() > 1.5
    return df


# ─── INSIGHTS ─────────────────────────────────────────────────────────────────

def get_insights(daily_df, app_df):
    insights = []
    if daily_df.empty or app_df.empty:
        return ["Not enough data yet!"]
    avg = daily_df["minutes"].mean()
    insights.append("Average daily screen time: " + str(int(avg//60)) + "h " + str(int(avg%60)) + "m")
    top = app_df.iloc[0]
    insights.append("Most used app: " + top['app'] + " — " + str(round(top['minutes']/60, 1)) + "h total")
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    daily_df = daily_df.dropna(subset=["date"])
    if daily_df.empty:
        return insights
    daily_df["is_weekend"] = daily_df["date"].dt.dayofweek >= 5
    we = daily_df[daily_df["is_weekend"]]["minutes"].mean()
    wd = daily_df[~daily_df["is_weekend"]]["minutes"].mean()
    if not pd.isna(we) and not pd.isna(wd):
        if we > wd * 1.2:
            insights.append("Screen time is higher on weekends (+" + str(round((we-wd)/60, 1)) + "h)")
        elif wd > we * 1.2:
            insights.append("Screen time is higher on weekdays (+" + str(round((wd-we)/60, 1)) + "h)")
        else:
            insights.append("Screen time is consistent across weekdays and weekends")
    if len(daily_df) >= 4:
        h1 = daily_df["minutes"].iloc[:len(daily_df)//2].mean()
        h2 = daily_df["minutes"].iloc[len(daily_df)//2:].mean()
        if h2 > h1 * 1.1:
            insights.append("Screen time has been increasing recently")
        elif h2 < h1 * 0.9:
            insights.append("Screen time has been decreasing — nice!")
        else:
            insights.append("Screen time has been stable recently")
    insights.append("Top 3 apps: " + ", ".join(app_df.head(3)['app'].tolist()))
    return insights


def get_social_media_mins(app_df):
    total = 0
    for _, row in app_df.iterrows():
        if any(s in row['app'].lower() for s in SOCIAL_APPS):
            total += row['minutes']
    return total


# ─── SLEEP IMPACT ─────────────────────────────────────────────────────────────

LATE_NIGHT_HOURS = {22, 23, 0, 1, 2}

SLEEP_RISK_LEVELS = {
    "Good":     {"color": "#34d399", "icon": "😴", "bar": 15},
    "Moderate": {"color": "#fbbf24", "icon": "😟", "bar": 50},
    "High":     {"color": "#fb923c", "icon": "😰", "bar": 75},
    "Severe":   {"color": "#f87171", "icon": "🚨", "bar": 100},
}


def get_sleep_impact(df):
    if df.empty or "hour" not in df.columns:
        return _empty_sleep_impact()

    valid_hour_df = df[df["hour"] >= 0]
    if valid_hour_df.empty:
        return _empty_sleep_impact()

    late_df = valid_hour_df[valid_hour_df["hour"].isin(LATE_NIGHT_HOURS)].copy()

    if late_df.empty:
        return {
            "risk_level": "Good",
            "color": SLEEP_RISK_LEVELS["Good"]["color"],
            "icon": SLEEP_RISK_LEVELS["Good"]["icon"],
            "bar_pct": SLEEP_RISK_LEVELS["Good"]["bar"],
            "nights_affected": 0,
            "total_late_minutes": 0,
            "avg_late_minutes": 0,
            "top_late_apps": [],
            "daily_late": [],
            "message": "No late-night screen usage detected. Great sleep hygiene!",
            "tips": ["Keep it up — avoid screens at least 1 hour before bed."]
        }

    nights_affected = late_df["date"].nunique()
    total_days = df["date"].nunique()
    total_late_mins = round(late_df["minutes"].sum())
    avg_late_mins = round(total_late_mins / max(nights_affected, 1))

    top_late = (
        late_df.groupby("app")["minutes"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    top_late_apps = [{"app": row["app"], "minutes": round(row["minutes"])}
                     for _, row in top_late.iterrows()]

    daily_late = (
        late_df.groupby("date")["minutes"]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    daily_late_list = [{"date": row["date"], "minutes": round(row["minutes"])}
                       for _, row in daily_late.iterrows()]

    nights_ratio = nights_affected / max(total_days, 1)
    risk_score = (avg_late_mins * 0.5) + (nights_ratio * 120)

    if risk_score < 20:
        level = "Good"
        message = "Minimal late-night usage. Your sleep is likely unaffected."
        tips = ["Try to keep screens off after 10 PM for optimal sleep."]
    elif risk_score < 50:
        level = "Moderate"
        message = ("You used screens late at night on " + str(nights_affected) +
                   " night(s), averaging " + str(avg_late_mins) +
                   " min. This may delay sleep onset by 15-30 minutes.")
        tips = ["Set a phone-down alarm at 10 PM.",
                "Switch to Night Mode / warm display after 9 PM.",
                "Replace late-night scrolling with reading a physical book."]
    elif risk_score < 100:
        level = "High"
        message = ("Late-night usage on " + str(nights_affected) +
                   " night(s) averaging " + str(avg_late_mins) +
                   " min. Blue light exposure at this level can reduce melatonin by up to 50%.")
        tips = ["Enable Do Not Disturb automatically at 10 PM.",
                "Remove social media apps from your home screen.",
                "Charge your phone outside the bedroom.",
                "Use blue-light blocking glasses after 8 PM."]
    else:
        level = "Severe"
        message = ("Severe late-night screen use detected — " + str(nights_affected) +
                   " night(s) with avg " + str(avg_late_mins) +
                   " min after 10 PM. This significantly disrupts your circadian rhythm.")
        tips = ["Implement a hard cutoff: no screens after 9:30 PM.",
                "Delete or log out of the most-used late-night apps.",
                "Use app blockers with a night schedule.",
                "Consider speaking to a doctor if this affects your daily functioning."]

    return {
        "risk_level": level,
        "color": SLEEP_RISK_LEVELS[level]["color"],
        "icon": SLEEP_RISK_LEVELS[level]["icon"],
        "bar_pct": SLEEP_RISK_LEVELS[level]["bar"],
        "nights_affected": nights_affected,
        "total_late_minutes": total_late_mins,
        "avg_late_minutes": avg_late_mins,
        "top_late_apps": top_late_apps,
        "daily_late": daily_late_list,
        "message": message,
        "tips": tips
    }


def _empty_sleep_impact():
    return {
        "risk_level": "Good",
        "color": SLEEP_RISK_LEVELS["Good"]["color"],
        "icon": SLEEP_RISK_LEVELS["Good"]["icon"],
        "bar_pct": 15,
        "nights_affected": 0,
        "total_late_minutes": 0,
        "avg_late_minutes": 0,
        "top_late_apps": [],
        "daily_late": [],
        "message": "No hourly data available. Make sure ActivityWatch is running.",
        "tips": ["Avoid screens at least 1 hour before bed for better sleep."]
    }