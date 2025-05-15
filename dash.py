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

    # Origin viz
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
    # Absolute completion heatmap
    st.subheader("Heatmap of Completions/Goals")
    fig2, ax2 = plt.subplots()
    sns.kdeplot(
        x=df_completions['recX'], y=df_completions['recY'],
        cmap="YlOrRd", shade=True, bw_adjust=1.2, ax=ax2
    )
    ax2.set_title("Completion/Goal Location Density")
    ax2.set_xlabel("Field X (width, meters)")
    ax2.set_ylabel("Field Y (length, meters)")
    st.pyplot(fig2)

    st.subheader("By Player: Relative Throw Maps & Debug for Goals")
    all_players = sorted(set(df['thrower'].unique()) | set(df['receiver'].dropna().unique()))
    selected_player = st.selectbox("Select player for individual heatmaps", all_players)

    player_throws = df[df['thrower'] == selected_player]
    player_catches = df_completions[df_completions['receiver'] == selected_player]
    st.write(f"🏆 {selected_player} caught **{player_catches['result'].eq('Goal').sum()}** goals")

    # Debug #2
    total_catches = len(player_catches)
    goal_count = int((player_catches['result'] == 'Goal').sum())
    st.write(f"🎯 **{selected_player}** has **{total_catches}** catch(es), of which **{goal_count}** {'is' if goal_count==1 else 'are'} Goal{'s' if goal_count!=1 else ''}.")

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

    import altair as alt

    # --- after you have `player_catches` and `player_throws` ---
    # build a DataFrame with one row per point (throw or catch)
    actions = player_catches.reset_index(drop=True)
    actions['id'] = actions.index  # unique per throw/catch pair

    # melt to long format
    df_plot = pd.concat([
        pd.DataFrame({
            'x': actions['thrX'],
            'y': actions['thrY'],
            'type': 'Throw Origin',
            'id': actions['id']
        }),
        pd.DataFrame({
            'x': actions['recX'],
            'y': actions['recY'],
            'type': actions['result'].map(lambda r: 'Goal' if r == 'Goal' else 'Catch'),
            'id': actions['id']
        })
    ], ignore_index=True)

    # define hover selection on the `id` field
    hover = alt.selection_single(
        fields=['id'],
        on='mouseover',
        empty='all',
        nearest=False
    )

    # build the chart
    chart = (
        alt.Chart(df_plot)
        .mark_point(filled=True, size=100)
        .encode(
            x=alt.X('x:Q', title='Field X (meters)'),
            y=alt.Y('y:Q', title='Field Y (meters)'),
            shape=alt.Shape('type:N',
                            scale=alt.Scale(domain=['Throw Origin', 'Catch', 'Goal'],
                                            range=['triangle-up', 'circle', 'square'])),
            color=alt.Color('id:N', legend=None),  # unique color per pair
            opacity=alt.condition(hover, alt.value(1), alt.value(0.2)),
            tooltip=['type:N', 'x:Q', 'y:Q']
        )
        .add_selection(hover)
        .properties(
            width=600,
            height=600,
            title=f"Catches & Throw Origins for {selected_player} (hover to highlight)"
        )
    )

    st.altair_chart(chart, use_container_width=True)

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
    # --- RELATIVE COMPLETION % HEATMAP WITH STATS ---
    st.subheader("🎯 Relative Completion % (by direction/length)")
    if not player_throws.empty:
        total_throws = len(player_throws)
        completions   = player_throws['result'].eq('Completion').sum()
        comp_pct      = completions / total_throws if total_throws > 0 else 0
        throw_distances = np.sqrt(
            (player_throws['thrX'] - player_throws['recX'])**2 +
            (player_throws['thrY'] - player_throws['recY'])**2
        )
        avg_throw_distance = throw_distances.mean()
        longest_throw      = throw_distances.max()

        st.info(
            f"**{selected_player}**\n"
            f"- Throws attempted: **{total_throws}**\n"
            f"- Completions: **{completions}** (Completion %: **{comp_pct:.1%}**)\n"
            f"- Average throw distance: **{avg_throw_distance:.1f} meters**\n"
            f"- Longest throw: **{longest_throw:.1f} meters**"
        )

        relX = player_throws['recX'] - player_throws['thrX']
        relY = player_throws['recY'] - player_throws['thrY']
        completed = player_throws['result'] == 'Completion'

        bin_count     = st.slider("Box size (bigger box = fewer bins)", min_value=3, max_value=15, value=8)
        debug_scatter = st.checkbox("Show individual throws on heatmap", value=False)

        heatmap, xedges, yedges = np.histogram2d(
            relX, relY, bins=bin_count, range=[[-40, 40], [-10, 70]]
        )
        completed_heatmap, _, _ = np.histogram2d(
            relX[completed], relY[completed], bins=[xedges, yedges]
        )
        completion_pct = np.where(heatmap > 0, completed_heatmap / heatmap, np.nan)

        fig, ax = plt.subplots(figsize=(6,6))
        cmap2  = plt.get_cmap("viridis")
        masked = np.ma.masked_where(heatmap == 0, completion_pct)
        norm   = mcolors.Normalize(vmin=0, vmax=1)
        c      = ax.pcolormesh(xedges, yedges, masked.T, cmap=cmap2, shading='auto', norm=norm)
        ax.plot(0, 0, 'ro', markersize=10, label='Origin')
        fig.colorbar(c, ax=ax, label="Completion %")
        ax.set_title(f"Completion %: {selected_player} (from origin)")
        ax.set_xlabel("Relative X (meters)")
        ax.set_ylabel("Relative Y (meters)")

        if debug_scatter:
            ax.scatter(relX[completed], relY[completed], color='lime', label='Completions', marker='o')
            ax.scatter(relX[~completed], relY[~completed], color='red', label='Throwaways', marker='x')
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
        labels={'direction':'Throw Direction','count':'Number of Throws'}
    ))

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
