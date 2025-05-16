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

# --- DATA LOAD & DEBUG #1 ---
uploaded_file = st.file_uploader("Upload your CSV (e.g., '2025-05-02-MAD-HTX.csv')", type="csv")
if not uploaded_file:
    st.info("👆 Upload your data to get started!")
    st.stop()

df = pd.read_csv(uploaded_file)

# Debug #1: list all result labels present in the file
st.write("🔍 All result types in this dataset:", df['result'].unique())

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

# --- INCLUDE COMPLETIONS & GOALS ---
goal_labels = ['Completion', 'Goal']
df_completions = df[df['result'].isin(goal_labels)]

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Throw Performance",
    "Field & Position",
    "Tactical Insights",
    "Team Metrics",
    "Player Impact"
])

# --- TAB 1: THROW PERFORMANCE ---
with tab1:
    st.markdown("### 📈 Throw-Level Performance\n_Summary of how individual players contribute to throwing and receiving in the selected dataset._")
    st.metric("Total Throws", len(df))
    st.metric("Total Completions/Goals", len(df_completions))
    st.divider()

    st.subheader("Completion Rate by Player (Thrower)")
    thrower_comp = df_completions.groupby('thrower').size()
    thrower_atts = df.groupby('thrower').size()
    thrower_pct = (thrower_comp / thrower_atts).fillna(0).sort_values(ascending=False)
    st.dataframe(thrower_pct.rename("Completion %").apply(lambda x: f"{x:.2%}"))

    st.subheader("Completion Rate by Player (Receiver)")
    receiver_comp = df_completions.groupby('receiver').size()
    receiver_atts = df_receivers.groupby('receiver').size()
    receiver_pct = (receiver_comp / receiver_atts).fillna(0).sort_values(ascending=False)
    st.dataframe(receiver_pct.rename("Completion %").apply(lambda x: f"{x:.2%}"))

    st.subheader("Top Thrower–Receiver Duos")
    duos = df_completions.groupby(['thrower','receiver']).size().sort_values(ascending=False)
    duos = duos.reset_index().rename(columns={0:'Completions/Goals'})
    st.dataframe(duos.head(15))

    st.divider()
    st.subheader("Most Active Throwers")
    st.bar_chart(thrower_atts.sort_values(ascending=False).head(15), use_container_width=True)

    st.subheader("Throwing Accuracy (Top 15)")
    st.bar_chart(thrower_pct.head(15), use_container_width=True)

# --- TAB 2: FIELD POSITION ANALYSIS ---
with tab2:
    st.markdown("### 🗺️ Field & Throw Position\n_See where throws originate and end, and analyze player tendencies spatially._")
    st.divider()

    # allow toggling raw points on/off for every density map
    overlay_points = st.checkbox("Overlay raw points on heatmaps", value=False)

    # 1) Origin visualization
    viz_option = st.radio(
        "Visualize throw origins as:",
        ["Heatmap", "Each Throw (Scatter)"], horizontal=True
    )
    fig, ax = plt.subplots()
    if viz_option == "Heatmap":
        sns.kdeplot(x=df['thrX'], y=df['thrY'], cmap="YlGnBu", shade=True, bw_adjust=1.2, ax=ax)
        if overlay_points:
            ax.scatter(df['thrX'], df['thrY'], c='navy', s=30, alpha=0.4, label='Throw origin')
            ax.legend(loc="upper right")
        ax.set_title("Throw Origin Density")
    else:
        ax.scatter(df['thrX'], df['thrY'], c='darkblue', alpha=0.6, edgecolor='white', s=70)
        ax.set_title("Each Throw Origin (Scatter Plot)")
    ax.set_xlabel("Field X (meters)")
    ax.set_ylabel("Field Y (meters)")
    st.pyplot(fig)

    st.divider()

    # 2) Catch/Goal 2D‐histogram heatmap
    st.subheader("📊 Catch/Goal Heatmap (2D Histogram)")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.histplot(
        x=df_completions['recX'], y=df_completions['recY'],
        bins=(15,10), cmap="YlOrRd", cbar=True,
        cbar_kws={'label':'Count'}, ax=ax2
    )
    if overlay_points:
        ax2.scatter(df_completions['recX'], df_completions['recY'],
                    c='black', s=20, alpha=0.3, label='Catch/Goal')
        ax2.legend(loc='upper right')
    ax2.set_title("Completion / Goal Location Counts")
    ax2.set_xlabel("Field X (meters)")
    ax2.set_ylabel("Field Y (meters)")
    ax2.set_aspect('equal', adjustable='box')
    st.pyplot(fig2)

    st.divider()
    # --- Scoring Probability Surface (Discrete Grid) ---
    st.divider()
    st.subheader("🎯 Predicted Scoring Probability by Catch Location")

    # 1) prepare your catch → goal labels
    df_catch = df.dropna(subset=['recX', 'recY']).copy()
    df_catch['is_goal'] = (df_catch['result'] == 'Goal').astype(int)

    # 2) train logistic regression
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(df_catch[['recX', 'recY']], df_catch['is_goal'])

    # 3) build a 60×60 mesh over the field
    x_min, x_max = df_catch['recX'].min(), df_catch['recX'].max()
    y_min, y_max = df_catch['recY'].min(), df_catch['recY'].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 60),
        np.linspace(y_min, y_max, 60)
    )

    # 4) predict score‐prob at each cell
    probs = model.predict_proba(
        np.c_[xx.ravel(), yy.ravel()]
    )[:, 1].reshape(xx.shape)

    # 5) plot with visible cell borders
    fig_prob, ax_prob = plt.subplots(figsize=(8, 6))
    mesh = ax_prob.pcolormesh(
        xx, yy, probs,
        cmap="RdYlGn_r",
        shading="flat",  # nearest‐neighbor colors
        edgecolors="lightgray",  # grid lines
        linewidth=0.5,
        vmin=0, vmax=1
    )
    fig_prob.colorbar(mesh, ax=ax_prob, label="P(Score)")
    ax_prob.set_title("Score Probability Surface")
    ax_prob.set_xlabel("Field X (m)")
    ax_prob.set_ylabel("Field Y (m)")
    ax_prob.set_aspect('equal', 'box')

    # 6) optional: overlay the raw catches for context
    if overlay_points:
        ax_prob.scatter(
            df_catch['recX'], df_catch['recY'],
            c='black', s=10, alpha=0.2
        )

    st.pyplot(fig_prob)

    # 3) Smoothed throwaway origin KDE
    st.subheader("🌫️ Smoothed Throwaway Origin Heatmap")
    drops = df[df['result']=='Throwaway']
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.kdeplot(
        x=drops['thrX'], y=drops['thrY'],
        cmap="Reds", shade=True,
        thresh=0.05, bw_adjust=1.0, levels=10, ax=ax3
    )
    if overlay_points:
        ax3.scatter(drops['thrX'], drops['thrY'],
                    c='darkred', s=40, alpha=0.5, label='Throwaway')
        ax3.legend(loc='upper right')
    ax3.set_title("Smoothed Density of Throwaway Origins")
    ax3.set_xlabel("Field X (meters)")
    ax3.set_ylabel("Field Y (meters)")
    ax3.set_aspect('equal', adjustable='box')
    st.pyplot(fig3)

    st.divider()

    # 4) By Player: relative maps + debug
    st.subheader("By Player: Relative Throw Maps & Debug for Goals")
    all_players = sorted(set(df['thrower'].unique()) | set(df['receiver'].dropna().unique()))
    selected_player = st.selectbox("Select player for individual heatmaps", all_players)

    player_throws = df[df['thrower']==selected_player]
    player_catches = df_completions[df_completions['receiver']==selected_player]

    # debug counts
    num_catches = len(player_catches)
    num_goals   = int((player_catches['result']=='Goal').sum())
    st.write(f"🎯 {selected_player} has {num_catches} catch(es), of which {num_goals} are Goals.")

    st.markdown("_Relative to each throw’s origin (0,0)_")
    # a) relative throw heatmap
    fig_rel_throw, ax_rel_throw = plt.subplots()
    if not player_throws.empty:
        relX = player_throws['recX'] - player_throws['thrX']
        relY = player_throws['recY'] - player_throws['thrY']
        sns.kdeplot(x=relX, y=relY, cmap="Blues", shade=True, bw_adjust=1.2, ax=ax_rel_throw)
        if overlay_points:
            ax_rel_throw.scatter(relX, relY, c='navy', s=30, alpha=0.4)
        ax_rel_throw.plot(0,0,'ro',markersize=8)
        ax_rel_throw.set_title(f"Throws by {selected_player} (from origin)")
        ax_rel_throw.set_xlabel("Relative X (m)")
        ax_rel_throw.set_ylabel("Relative Y (m)")
        st.pyplot(fig_rel_throw)
    else:
        st.info("No throws for this player.")

    # b) relative catch heatmap
    fig_rel_catch, ax_rel_catch = plt.subplots()
    if not player_catches.empty:
        relX_c = player_catches['recX'] - player_catches['thrX']
        relY_c = player_catches['recY'] - player_catches['thrY']
        sns.kdeplot(x=relX_c, y=relY_c, cmap="Greens", shade=True, bw_adjust=1.2, ax=ax_rel_catch)
        if overlay_points:
            ax_rel_catch.scatter(relX_c, relY_c, c='darkgreen', s=30, alpha=0.4)
        ax_rel_catch.plot(0,0,'ro',markersize=8)
        ax_rel_catch.set_title(f"Catches by {selected_player} (from origin)")
        ax_rel_catch.set_xlabel("Relative X (m)")
        ax_rel_catch.set_ylabel("Relative Y (m)")
        st.pyplot(fig_rel_catch)
    else:
        st.info("No catches for this player.")

    st.divider()

    # 5) Relative completion % heatmap
    st.subheader("🎯 Relative Completion % (by direction/length)")
    if not player_throws.empty:
        total = len(player_throws)
        comp  = player_throws['result'].eq('Completion').sum()
        pct   = comp/total if total>0 else 0
        st.info(f"- Attempts: {total}  •  Completions: {comp} ({pct:.1%})")

        relX = player_throws['recX'] - player_throws['thrX']
        relY = player_throws['recY'] - player_throws['thrY']
        completed = player_throws['result']=='Completion'

        bins = st.slider("Bins per axis",3,15,8)
        heatmap, xedges, yedges = np.histogram2d(relX, relY, bins=bins, range=[[-40,40],[-10,70]])
        compmap, _, _ = np.histogram2d(relX[completed], relY[completed], bins=[xedges,yedges])
        pctmap = np.where(heatmap>0, compmap/heatmap, np.nan)

        fig_pct, ax_pct = plt.subplots(figsize=(6,6))
        cmap = plt.get_cmap("viridis")
        mesh = ax_pct.pcolormesh(xedges, yedges, pctmap.T, cmap=cmap, shading='auto', vmin=0, vmax=1)
        fig_pct.colorbar(mesh, ax=ax_pct, label="Completion %")
        if overlay_points:
            ax_pct.scatter(relX[completed], relY[completed], c='lime', s=30, alpha=0.5)
            ax_pct.scatter(relX[~completed], relY[~completed], c='red', s=30, alpha=0.5)
        ax_pct.plot(0,0,'ro',markersize=8)
        ax_pct.set_title(f"Completion % for {selected_player}")
        ax_pct.set_xlabel("Relative X (m)")
        ax_pct.set_ylabel("Relative Y (m)")
        st.pyplot(fig_pct)
    else:
        st.info("No throws recorded for this player.")

    st.divider()

    # 6) Throw distance distribution & probability
    st.subheader("Throw Distance Distribution")
    df['distance'] = np.sqrt((df['thrX']-df['recX'])**2 + (df['thrY']-df['recY'])**2)
    st.plotly_chart(px.histogram(df, x='distance', nbins=30, labels={'distance':'Throw Distance (m)'}, title="Distance Distribution"))

    st.subheader("Completion Probability by Distance")
    bins = np.arange(0, df['distance'].max()+5, 5)
    df['dist_bin'] = pd.cut(df['distance'], bins)
    prob = df.groupby('dist_bin')['result'].apply(lambda x: (x=='Completion').mean())
    mids = prob.index.map(lambda x: x.mid)
    st.line_chart(pd.DataFrame({'Distance':mids,'Prob':prob.values}).set_index('Distance'))

    st.subheader("Throw Direction Analysis")
    df['direction'] = np.where(df['recY']>df['thrY'],'Forward','Backward/Other')
    dir_counts = df.groupby('direction')['result'].value_counts().unstack().fillna(0)
    st.dataframe(dir_counts)
    st.plotly_chart(px.histogram(df, x='direction', color='result', barmode='group', title="Throw Direction Outcomes"))


# --- TAB 3: TACTICAL INSIGHTS ---
with tab3:
    st.markdown("### 🧠 Tactical Insights\n_Analyze turnovers, risky zones, and compare play styles._")
    st.subheader("Throw Outcomes")
    st.plotly_chart(px.histogram(
        df, x='result', color='result', title="Throw Outcomes",
        labels={'result':'Throw Result','count':'Number of Throws'}
    ))
    st.divider()

    st.subheader("Turnover Distances")
    df_turnover = df[df['result'] != 'Completion']
    st.plotly_chart(px.histogram(
        df_turnover, x='distance', nbins=20,
        title="Turnover Throw Distances",
        labels={'distance':'Throw Distance (meters)','count':'Number of Turnovers'}
    ))

    st.subheader("Field Zones with Highest Drop Rates")
    drops = df[df['result'] == 'Drop']
    if not drops.empty:
        fig3, ax3 = plt.subplots()
        sns.kdeplot(x=drops['recX'], y=drops['recY'], cmap="Reds", shade=True, bw_adjust=1.2, ax=ax3)
        ax3.set_title("Drop Locations Heatmap")
        ax3.set_xlabel("Field X (width, meters)")
        ax3.set_ylabel("Field Y (length, meters)")
        st.pyplot(fig3)
    else:
        st.info("No drops in data.")

    st.divider()
    st.subheader("Play Style Comparison by Game")
    st.dataframe(df['gameID'].value_counts())

# --- TAB 4: TEAM METRICS ---
with tab4:
    st.markdown("### 🔄 Team Metrics\n_See how efficiently the team moves the disc and scores points._")
    st.subheader("Completions per Possession")
    poss_eff = df.groupby('point').apply(lambda x: (x['result'] == 'Completion').sum())
    st.line_chart(poss_eff.rename("Completions per Possession"))

    st.subheader("Throws per Point")
    throws_per_point = df.groupby('point').size()
    st.line_chart(throws_per_point.rename("Throws per Point"))

    st.subheader("Scoring Efficiency by Throw Count")
    point_scores = df.groupby('point').tail(1).reset_index()
    scoring = point_scores['result'] == 'Completion'
    throw_counts = throws_per_point
    st.scatter_chart(pd.DataFrame({"Throws": throw_counts, "Scored": scoring.values.astype(int)}))

# --- TAB 5: PLAYER IMPACT ---
with tab5:
    st.markdown("### 🌟 Player Impact\n_Who's making a difference with big yards, scores, or clutch throws?_")
    st.subheader("Yards Gained per Throw (Top 10)")
    yards = df[df['result']=='Completion'].copy()
    yards['yards'] = np.sqrt((yards['thrX']-yards['recX'])**2 + (yards['thrY']-yards['recY'])**2) * 1.09361
    top_yards = yards.groupby('thrower')['yards'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_yards, use_container_width=True)

    st.subheader("Scoring Involvement")
    last_throws = df.groupby('point').tail(1)
    inv = pd.concat([last_throws['thrower'], last_throws['receiver']]).value_counts()
    st.bar_chart(inv.head(10), use_container_width=True)

    st.subheader("Clutch Throws")
    max_point = df['point'].max()
    clutch_points = df[df['point'] >= max_point-3]
    clutch = clutch_points.groupby('thrower').size().sort_values(ascending=False)
    st.bar_chart(clutch.head(10), use_container_width=True)
