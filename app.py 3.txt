import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Instagram Analytics Dashboard", layout="wide")
st.title("📸 Instagram User Engagement & Content Performance Analytics")

# ---------------------------------------------------
# 1️⃣ Generate Synthetic Dataset (1000 rows)
# ---------------------------------------------------

def generate_instagram_data():
    np.random.seed(42)

    content_types = ["Image", "Video", "Reel", "Carousel"]
    captions = [
        "New product launch! #launch #brand",
        "Behind the scenes #bts #team",
        "Flash sale today! #sale #offer",
        "Motivation Monday #mondaymotivation",
        "Travel diaries #wanderlust #travel"
    ]

    dates = pd.date_range("2024-01-01", periods=1000, freq="H")

    df = pd.DataFrame({
        "post_id": range(1, 1001),
        "content_type": np.random.choice(content_types, 1000),
        "post_datetime": np.random.choice(dates, 1000),
        "likes": np.random.randint(100, 5000, 1000),
        "comments": np.random.randint(10, 500, 1000),
        "shares": np.random.randint(5, 300, 1000),
        "followers_at_post": np.random.randint(5000, 20000, 1000),
        "caption": np.random.choice(captions, 1000)
    })

    return df

# ---------------------------------------------------
# 2️⃣ Upload Dataset
# ---------------------------------------------------

uploaded_file = st.file_uploader("Upload Instagram Dataset (CSV) or use generated data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = generate_instagram_data()

# ---------------------------------------------------
# 3️⃣ Data Cleaning
# ---------------------------------------------------

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df["post_datetime"] = pd.to_datetime(df["post_datetime"])

# ---------------------------------------------------
# 4️⃣ Regex: Extract Hashtags
# ---------------------------------------------------

df["hashtags"] = df["caption"].apply(lambda x: re.findall(r"#(\w+)", x))
df["hashtag_count"] = df["hashtags"].apply(len)

# ---------------------------------------------------
# 5️⃣ Feature Engineering
# ---------------------------------------------------

df["engagement"] = df["likes"] + df["comments"] + df["shares"]
df["engagement_rate"] = df["engagement"] / df["followers_at_post"]

df["hour"] = df["post_datetime"].dt.hour
df["day_of_week"] = df["post_datetime"].dt.day_name()
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

# ---------------------------------------------------
# 6️⃣ Normalization
# ---------------------------------------------------

scaler = MinMaxScaler()
df[["engagement_norm"]] = scaler.fit_transform(df[["engagement"]])

# ---------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------

content_filter = st.sidebar.multiselect(
    "Select Content Type",
    df["content_type"].unique(),
    default=df["content_type"].unique()
)

df = df[df["content_type"].isin(content_filter)]

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Posts", len(df))
col2.metric("Avg Engagement", int(df["engagement"].mean()))
col3.metric("Avg Engagement Rate", round(df["engagement_rate"].mean(), 3))
col4.metric("Avg Hashtags Used", round(df["hashtag_count"].mean(), 2))

st.markdown("---")

# ---------------------------------------------------
# 1️⃣ Which Content Type Performs Best?
# ---------------------------------------------------

st.subheader("🏆 Engagement by Content Type")

content_perf = df.groupby("content_type")["engagement"].mean().sort_values(ascending=False)
st.write(content_perf)

fig1, ax1 = plt.subplots()
content_perf.plot(kind="bar", ax=ax1)
st.pyplot(fig1)

# ---------------------------------------------------
# 2️⃣ Are Videos Better Than Images? (Hypothesis Test)
# ---------------------------------------------------

st.subheader("📊 Video vs Image Engagement (T-Test)")

video = df[df["content_type"] == "Video"]["engagement"]
image = df[df["content_type"] == "Image"]["engagement"]

if len(video) > 0 and len(image) > 0:
    t_stat, p_value = stats.ttest_ind(video, image)

    st.write("T-Statistic:", round(t_stat, 3))
    st.write("P-Value:", round(p_value, 4))

    if p_value < 0.05:
        st.success("Videos and Images perform significantly differently!")
    else:
        st.info("No significant difference between Videos and Images.")
else:
    st.warning("Not enough data for comparison.")

# ---------------------------------------------------
# 3️⃣ Best Posting Time
# ---------------------------------------------------

st.subheader("⏰ Engagement by Hour")

hour_perf = df.groupby("hour")["engagement"].mean()

fig2, ax2 = plt.subplots()
hour_perf.plot(ax=ax2)
st.pyplot(fig2)

# ---------------------------------------------------
# 4️⃣ Hashtag Impact
# ---------------------------------------------------

st.subheader("🔖 Hashtag Count vs Engagement")

fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x="hashtag_count", y="engagement", ax=ax3)
st.pyplot(fig3)

correlation = df["hashtag_count"].corr(df["engagement"])
st.write("Correlation:", round(correlation, 2))

# ---------------------------------------------------
# 5️⃣ Posts That Should Be Boosted
# ---------------------------------------------------

st.subheader("🚀 High Potential Posts (High Engagement Rate)")

boost_posts = df.sort_values("engagement_rate", ascending=False).head(10)
st.dataframe(boost_posts[["post]()]()_
