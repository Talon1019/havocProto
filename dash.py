import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.colors as mcolors

st.set_page_config(page_title="Ultimate Frisbee Throw Dashboard", layout="wide")

st.title("Ultimate Frisbee Throw Analytics Dashboard")
st.markdown(
    """
    Upload your throw-level CSV (e.g., '2025-05-02-MAD-HTX.csv') to get started.

    Use the dropdown below to select which game you want to analyze, or select "All Games" for full-season analysis.
    """
)

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)

# ---- GAME FILTER ----
game_ids = df['gameID'].unique().tolist()
game_option = st.selectbox(
    "Choose a game to analyze",
    options=["All Games"] + game_ids,
    help="Filter all dashboard charts/tables to one specific game, or view all games together."
)

if game_option != "All Games":
    df = df[df['gameID'] == game_option]

df_receivers = df.dropna(subset=['receiver'])
df_completions = df[df['result'] == 'Completion']

# --- TAB LAYOUT ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Throw-Level Performance",
    "Field Position Analysis",
    "Tactical Insights",
    "Team Metrics",
    "Player Impact"
])

# ========== 1. THROW-LEVEL PERFORMANCE ==========
with tab1:
    st.header("🔁 Throw-Level Performance")
    st.caption("Understand player involvement, completion rates, and most common throw connections for this game.")

    st.subheader("Completion Rate by Player (Thrower)")
    st.caption("The percentage of passes thrown by each player that resulted in completions.")
    thrower_comp = df_completions.groupby('thrower').size()
    thrower_atts = df.groupby('thrower').size()
    thrower_pct = (thrower_comp / thrower_atts).fillna(0).sort_values(ascending=False)
    st.dataframe(thrower_pct.rename("Completion %").apply(lambda x: f"{x:.2%}"))

    st.subheader("Completion Rate by Player (Receiver)")
    st.caption("The percentage of targets each player received that were completed.")
    receiver_comp = df_completions.groupby('receiver').size()
    receiver_atts = df_receivers.groupby('receiver').size()
    receiver_pct = (receiver_comp / receiver_atts).fillna(0).sort_values(ascending=False)
    st.dataframe(receiver_pct.rename("Completion %").apply(lambda x: f"{x:.2%}"))

    st.subheader("Most Frequent Thrower–Receiver Duos")
    st.caption("Who connected most often? Top duos by completed passes.")
    duos = df_completions.groupby(['thrower','receiver']).size().sort_values(ascending=False)
    duos = duos.reset_index().rename(columns={0:'Completions'})
    st.dataframe(duos.head(15))

    st.subheader("Most Active Players by Throw Count")
    st.caption("Which players threw the most passes?")
    st.bar_chart(thrower_atts.sort_values(ascending=False).head(15), use_container_width=True)

    st.subheader("Throwing Accuracy (Completion % by Player)")
    st.caption("Top 15 players by throw completion percentage (minimum 1 attempt).")
    st.bar_chart(thrower_pct.head(15), use_container_width=True)

# ========== 2. FIELD POSITION ANALYSIS ==========
with tab2:
    st.header("🧭 Field Position Analysis")
    st.caption("Analyze where throws start and end, and which field zones are most efficient.")

    st.subheader("Heatmap of Throw Origins")
    st.caption("Where on the field do most throws start?")
    fig, ax = plt.subplots()
    sns.kdeplot(
        x=df['thrX'], y=df['thrY'],
        cmap="YlGnBu", shade=True, bw_adjust=1.2, ax=ax)
    ax.set_title("Throw Origin Density")
    ax.set_xlabel("Field X position (width, meters)")
    ax.set_ylabel("Field Y position (length, meters)")
    st.pyplot(fig)

    st.subheader("Heatmap of Completions")
    st.caption("Where do completed passes most often land on the field?")
    fig2, ax2 = plt.subplots()
    sns.kdeplot(
        x=df_completions['recX'], y=df_completions['recY'],
        cmap="YlOrRd", shade=True, bw_adjust=1.2, ax=ax2)
    ax2.set_title("Completion Location Density")
    ax2.set_xlabel("Field X position (width, meters)")
    ax2.set_ylabel("Field Y position (length, meters)")
    st.pyplot(fig2)

    st.subheader("📊 Relative Heatmaps by Player (All Throws Start at Origin)")
    st.caption("Throw and catch patterns for the selected player, relative to the throw start (0,0). Useful for analyzing typical throw lengths and directions.")

    # Get list of players
    all_players = sorted(set(df['thrower'].unique()) | set(df['receiver'].dropna().unique()))
    selected_player = st.selectbox(
        "Select a player for relative heatmaps",
        all_players,
        key="player_heatmap"
    )

    # --- Relative Throw Heatmap ---
    player_throws = df[df['thrower'] == selected_player]
    fig_pthrow_rel, ax_pthrow_rel = plt.subplots()
    if not player_throws.empty:
        rel_recX = player_throws['recX'] - player_throws['thrX']
        rel_recY = player_throws['recY'] - player_throws['thrY']
        sns.kdeplot(
            x=rel_recX, y=rel_recY,
            cmap="Blues", shade=True, bw_adjust=1.2, ax=ax_pthrow_rel
        )
        ax_pthrow_rel.plot(0, 0, 'ro', markersize=10, label='Origin (Throw Start)')
        ax_pthrow_rel.set_title(f"Relative Throw Endpoints (Throws from (0,0)): {selected_player}")
        ax_pthrow_rel.set_xlabel("Relative X (meters)")
        ax_pthrow_rel.set_ylabel("Relative Y (meters)")
        ax_pthrow_rel.legend()
        st.pyplot(fig_pthrow_rel)
    else:
        st.info(f"No throws recorded for {selected_player}.")

    # --- Relative Catch Heatmap (only completions) ---
    player_catches = df_completions[df_completions['receiver'] == selected_player]
    fig_pcatch_rel, ax_pcatch_rel = plt.subplots()
    if not player_catches.empty:
        rel_recX_c = player_catches['recX'] - player_catches['thrX']
        rel_recY_c = player_catches['recY'] - player_catches['thrY']
        sns.kdeplot(
            x=rel_recX_c, y=rel_recY_c,
            cmap="Greens", shade=True, bw_adjust=1.2, ax=ax_pcatch_rel
        )
        ax_pcatch_rel.plot(0, 0, 'ro', markersize=10, label='Origin (Throw Start)')
        ax_pcatch_rel.set_title(f"Relative Catch Locations (Catches from (0,0)): {selected_player}")
        ax_pcatch_rel.set_xlabel("Relative X (meters)")
        ax_pcatch_rel.set_ylabel("Relative Y (meters)")
        ax_pcatch_rel.legend()
        st.pyplot(fig_pcatch_rel)
    else:
        st.info(f"No completions caught by {selected_player}.")

    # --- Relative Completion % Heatmap ---
    st.subheader("🎯 Relative Completion Percentage Heatmap (by throw direction/length)")
    st.caption("Shows how likely throws of a given direction and distance are to be completed, for the selected player. Each square shows completion % for throws ending in that area.")
    if not player_throws.empty:
        relX = player_throws['recX'] - player_throws['thrX']
        relY = player_throws['recY'] - player_throws['thrY']
        completed = player_throws['result'] == 'Completion'

        bins = 30  # Adjust for resolution
        heatmap, xedges, yedges = np.histogram2d(
            relX, relY, bins=bins, range=[[-40, 40], [-10, 70]]
        )
        completed_heatmap, _, _ = np.histogram2d(
            relX[completed], relY[completed], bins=[xedges, yedges]
        )
        completion_pct = np.where(heatmap > 0, completed_heatmap / heatmap, np.nan)
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.get_cmap("viridis")
        masked = np.ma.masked_invalid(completion_pct)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        c = ax.pcolormesh(
            xedges, yedges, masked.T, cmap=cmap, shading='auto', norm=norm
        )
        ax.plot(0, 0, 'ro', markersize=10, label='Origin (Throw Start)')
        fig.colorbar(c, ax=ax, label="Completion %")
        ax.set_title(f"Relative Completion % (Throws from (0,0)): {selected_player}")
        ax.set_xlabel("Relative X (meters)")
        ax.set_ylabel("Relative Y (meters)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info(f"No throws recorded for {selected_player}.")

    st.subheader("Throw Distance Distribution")
    st.caption("Histogram of throw distances (in meters) across all points in the filtered data.")
    df['distance'] = np.sqrt((df['thrX']-df['recX'])**2 + (df['thrY']-df['recY'])**2)
    st.plotly_chart(px.histogram(
        df, x='distance', nbins=30,
        title="Throw Distance Distribution",
        labels={'distance': 'Throw Distance (meters)', 'count': 'Number of Throws'}
    ))

    st.subheader("Average Completion Probability by Distance")
    st.caption("Line chart showing the chance of completion for throws of different lengths (binned every 5 meters).")
    bins = np.arange(0, df['distance'].max() + 5, 5)
    df['distance_bin'] = pd.cut(df['distance'], bins)
    prob_by_dist = df.groupby('distance_bin')['result'].apply(lambda x: (x == 'Completion').mean())
    bin_midpoints = prob_by_dist.index.map(lambda x: x.mid)
    prob_by_dist_df = pd.DataFrame({
        "Distance": bin_midpoints,
        "Completion Probability": prob_by_dist.values
    })
    st.line_chart(prob_by_dist_df.set_index("Distance"))

    st.subheader("Throw Direction Analysis")
    st.caption("Compares results of throws that move the disc forward (downfield) versus backward/other directions.")
    df['direction'] = np.where(df['recY'] > df['thrY'], 'Forward', 'Backward/Other')
    dir_counts = df.groupby('direction')['result'].value_counts().unstack().fillna(0)
    st.dataframe(dir_counts)
    st.plotly_chart(px.histogram(
        df, x='direction', color='result', barmode='group',
        title="Throw Direction Outcomes",
        labels={'direction': 'Throw Direction', 'count': 'Number of Throws'}
    ))

# ========== 3. TACTICAL INSIGHTS ==========
with tab3:
    st.header("🎯 Tactical Insights")
    st.caption("Dig into turnovers, risky throws, and field zones that may need improvement.")

    st.subheader("Throw Types by Result")
    st.caption("Distribution of throw outcomes (completion, drop, etc.).")
    st.plotly_chart(px.histogram(
        df, x='result', color='result', title="Throw Outcomes",
        labels={'result': 'Throw Result', 'count': 'Number of Throws'}
    ))

    st.subheader("Common Throw Lengths Leading to Turnovers")
    st.caption("Histogram showing which throw lengths most often lead to turnovers.")
    df_turnover = df[df['result'] != 'Completion']
    st.plotly_chart(px.histogram(
        df_turnover, x='distance', nbins=20,
        title="Turnover Throw Distances",
        labels={'distance': 'Throw Distance (meters)', 'count': 'Number of Turnovers'}
    ))

    st.subheader("Field Zones with Highest Drop Rates")
    st.caption("Heatmap showing where on the field drops most often occur.")
    drops = df[df['result'] == 'Drop']
    if not drops.empty:
        fig3, ax3 = plt.subplots()
        sns.kdeplot(
            x=drops['recX'], y=drops['recY'],
            cmap="Reds", shade=True, bw_adjust=1.2, ax=ax3)
        ax3.set_title("Drop Locations Heatmap")
        ax3.set_xlabel("Field X position (width, meters)")
        ax3.set_ylabel("Field Y position (length, meters)")
        st.pyplot(fig3)
    else:
        st.info("No drops in data.")

    st.subheader("Play Style Comparison (by Team/Game)")
    st.caption("Games present in the dataset and their total number of throws.")
    st.dataframe(df['gameID'].value_counts())

# ========== 4. TEAM-LEVEL METRICS ==========
with tab4:
    st.header("🔄 Team-Level Metrics")
    st.caption("How well does the team move the disc and convert possessions into points?")

    st.subheader("Possession Efficiency (Completions per Possession)")
    st.caption("Number of completions made during each possession/point.")
    poss_eff = df.groupby('point').apply(lambda x: (x['result']=='Completion').sum())
    st.line_chart(poss_eff.rename("Completions per Possession"))

    st.subheader("Number of Throws per Point")
    st.caption("How many passes are made on each point?")
    throws_per_point = df.groupby('point').size()
    st.line_chart(throws_per_point.rename("Throws per Point"))

    st.subheader("Scoring Efficiency by Throw Count")
    st.caption("Shows how often possessions of a given length (number of throws) result in a score.")
    # Assume last row of each point is a score if completion, else not
    point_scores = df.groupby('point').tail(1).reset_index()
    scoring = point_scores['result'] == 'Completion'
    throw_counts = throws_per_point
    st.scatter_chart(pd.DataFrame({"Throws": throw_counts, "Scored": scoring.values.astype(int)}))

# ========== 5. PLAYER IMPACT METRICS ==========
with tab5:
    st.header("📈 Player Impact Metrics")
    st.caption("See who contributed most to yardage, scores, and clutch moments.")

    st.subheader("Yards Gained per Throw (Top 10)")
    st.caption("Average yards gained per completed throw, by thrower.")
    yards = df[df['result']=='Completion'].copy()
    yards['yards'] = np.sqrt((yards['thrX']-yards['recX'])**2 + (yards['thrY']-yards['recY'])**2) * 1.09361  # Meters to yards
    top_yards = yards.groupby('thrower')['yards'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_yards, use_container_width=True)

    st.subheader("Involvement Rate in Scoring Points")
    st.caption("Counts how often a player was the thrower or receiver on a scoring play.")
    last_throws = df.groupby('point').tail(1)
    inv = pd.concat([last_throws['thrower'], last_throws['receiver']]).value_counts()
    st.bar_chart(inv.head(10), use_container_width=True)

    st.subheader("Clutch Throws (Late or Tied Points)")
    st.caption("Throws made during the final three points, often high-pressure situations.")
    max_point = df['point'].max()
    clutch_points = df[df['point'] >= max_point-3]
    clutch = clutch_points.groupby('thrower').size().sort_values(ascending=False)
    st.bar_chart(clutch.head(10), use_container_width=True)
