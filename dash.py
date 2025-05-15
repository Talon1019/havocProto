import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.colors as mcolors

st.set_page_config(page_title="Ultimate Frisbee Throw Dashboard", layout="wide")

# --- APP INTRO ---
st.title("Ultimate Frisbee Throw Analytics Dashboard")
st.markdown("""
**Welcome!**  
This dashboard helps players and coaches analyze throw-by-throw ultimate data, player tendencies, and team strategies.
- 📊 **Choose a game** or see all games at once.
- 🏷️ **Switch tabs** above to explore throw performance, field position, tactics, team stats, and player impact.
- **Hover/click charts** for more detail. See "By Player" sections for personalized maps.
""")

uploaded_file = st.file_uploader("Upload your CSV (e.g., '2025-05-02-MAD-HTX.csv')", type="csv")
if not uploaded_file:
    st.info("👆 Upload your data to get started!")
    st.stop()

df = pd.read_csv(uploaded_file)

# --- GAME SELECTION ---
game_ids = df['gameID'].unique().tolist()
game_option = st.selectbox(
    "1️⃣ Filter by Game (optional):",
    options=["All Games"] + game_ids,
    help="Analyze one game or all games together."
)
if game_option != "All Games":
    df = df[df['gameID'] == game_option]

df_receivers = df.dropna(subset=['receiver'])
df_completions = df[df['result'] == 'Completion']

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Throw Performance",
    "Field & Position",
    "Tactical Insights",
    "Team Metrics",
    "Player Impact"
])

# --- TAB 2: FIELD POSITION ANALYSIS ---
with tab2:
    st.markdown("### 🗺️ Field & Throw Position\n_See where throws originate and end, and analyze player tendencies spatially._")
    st.divider()

    viz_option = st.radio(
        "Visualize throw origins as:",
        ["Heatmap", "Each Throw (Scatter)"], horizontal=True
    )
    fig, ax = plt.subplots()
    if viz_option == "Heatmap":
        sns.kdeplot(x=df['thrX'], y=df['thrY'], cmap="YlGnBu", shade=True, bw_adjust=1.2, ax=ax)
        ax.set_title("Throw Origin Density")
    else:
        ax.scatter(df['thrX'], df['thrY'], c='darkblue', alpha=0.6, edgecolor='white', s=70)
        ax.set_title("Each Throw Origin (Scatter Plot)")
    ax.set_xlabel("Field X (width, meters)")
    ax.set_ylabel("Field Y (length, meters)")
    st.pyplot(fig)
    st.divider()

    st.subheader("Heatmap of Completions")
    fig2, ax2 = plt.subplots()
    sns.kdeplot(
        x=df_completions['recX'], y=df_completions['recY'],
        cmap="YlOrRd", shade=True, bw_adjust=1.2, ax=ax2)
    ax2.set_title("Completion Location Density")
    ax2.set_xlabel("Field X (width, meters)")
    ax2.set_ylabel("Field Y (length, meters)")
    st.pyplot(fig2)

    st.subheader("By Player: Relative Throw Maps")
    all_players = sorted(set(df['thrower'].unique()) | set(df['receiver'].dropna().unique()))
    selected_player = st.selectbox("Select player for individual heatmaps", all_players)
    player_throws = df[df['thrower'] == selected_player]
    player_catches = df_completions[df_completions['receiver'] == selected_player]
    st.markdown("_Visualizes throws/catches relative to the throw start (origin at (0,0))_")

    # Relative throw heatmap
    fig_pthrow_rel, ax_pthrow_rel = plt.subplots()
    if not player_throws.empty:
        rel_recX = player_throws['recX'] - player_throws['thrX']
        rel_recY = player_throws['recY'] - player_throws['thrY']
        sns.kdeplot(x=rel_recX, y=rel_recY, cmap="Blues", shade=True, bw_adjust=1.2, ax=ax_pthrow_rel)
        ax_pthrow_rel.plot(0, 0, 'ro', markersize=10, label='Origin')
        ax_pthrow_rel.set_title(f"Throws by {selected_player} (from origin)")
        ax_pthrow_rel.set_xlabel("Relative X (meters)")
        ax_pthrow_rel.set_ylabel("Relative Y (meters)")
        ax_pthrow_rel.legend()
        st.pyplot(fig_pthrow_rel)
    else:
        st.info(f"No throws for {selected_player}")

    # Relative catch heatmap
    fig_pcatch_rel, ax_pcatch_rel = plt.subplots()
    if not player_catches.empty:
        rel_recX_c = player_catches['recX'] - player_catches['thrX']
        rel_recY_c = player_catches['recY'] - player_catches['thrY']
        sns.kdeplot(x=rel_recX_c, y=rel_recY_c, cmap="Greens", shade=True, bw_adjust=1.2, ax=ax_pcatch_rel)
        ax_pcatch_rel.plot(0, 0, 'ro', markersize=10, label='Origin')
        ax_pcatch_rel.set_title(f"Catches by {selected_player} (from origin)")
        ax_pcatch_rel.set_xlabel("Relative X (meters)")
        ax_pcatch_rel.set_ylabel("Relative Y (meters)")
        ax_pcatch_rel.legend()
        st.pyplot(fig_pcatch_rel)
    else:
        st.info(f"No completions caught by {selected_player}")

    st.divider()
    # Relative completion percentage heatmap WITH PLAYER INFO
    st.subheader("🎯 Relative Completion % (by direction/length)")
    if not player_throws.empty:
        # ---- PLAYER INFO ----
        total_throws = len(player_throws)
        completions = player_throws['result'].eq('Completion').sum()
        comp_pct = completions / total_throws if total_throws > 0 else 0
        avg_throw_distance = np.sqrt(
            (player_throws['thrX'] - player_throws['recX'])**2 +
            (player_throws['thrY'] - player_throws['recY'])**2
        ).mean()
        longest_throw = np.sqrt(
            (player_throws['thrX'] - player_throws['recX'])**2 +
            (player_throws['thrY'] - player_throws['recY'])**2
        ).max()
        st.info(
            f"**{selected_player}**\n"
            f"- Throws attempted: **{total_throws}**\n"
            f"- Completions: **{completions}**  (Completion %: **{comp_pct:.1%}**)\n"
            f"- Average throw distance: **{avg_throw_distance:.1f} meters**\n"
            f"- Longest throw: **{longest_throw:.1f} meters**"
        )

        bin_count = st.slider(
            "Box size (bigger box = fewer bins)", min_value=5, max_value=30, value=12
        )
        relX = player_throws['recX'] - player_throws['thrX']
        relY = player_throws['recY'] - player_throws['thrY']
        completed = player_throws['result'] == 'Completion'
        heatmap, xedges, yedges = np.histogram2d(
            relX, relY, bins=bin_count, range=[[-40, 40], [-10, 70]]
        )
        completed_heatmap, _, _ = np.histogram2d(
            relX[completed], relY[completed], bins=[xedges, yedges]
        )
        completion_pct = np.where(heatmap > 0, completed_heatmap / heatmap, np.nan)
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.get_cmap("viridis")
        masked = np.ma.masked_where(heatmap == 0, completion_pct)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        c = ax.pcolormesh(xedges, yedges, masked.T, cmap=cmap, shading='auto', norm=norm)
        ax.plot(0, 0, 'ro', markersize=10, label='Origin')
        fig.colorbar(c, ax=ax, label="Completion %")
        ax.set_title(f"Completion %: {selected_player} (from origin)")
        ax.set_xlabel("Relative X (meters)")
        ax.set_ylabel("Relative Y (meters)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info(f"No throws recorded for {selected_player}.")

    st.divider()
    st.subheader("Throw Distance Distribution")
    df['distance'] = np.sqrt((df['thrX']-df['recX'])**2 + (df['thrY']-df['recY'])**2)
    st.plotly_chart(px.histogram(
        df, x='distance', nbins=30,
        title="Throw Distance Distribution",
        labels={'distance': 'Throw Distance (meters)', 'count': 'Number of Throws'}
    ))

    st.subheader("Completion Probability by Distance")
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
    df['direction'] = np.where(df['recY'] > df['thrY'], 'Forward', 'Backward/Other')
    dir_counts = df.groupby('direction')['result'].value_counts().unstack().fillna(0)
    st.dataframe(dir_counts)
    st.plotly_chart(px.histogram(
        df, x='direction', color='result', barmode='group',
        title="Throw Direction Outcomes",
        labels={'direction': 'Throw Direction', 'count': 'Number of Throws'}
    ))
