"""
Full 5-Video Head-Tracking Analysis
Processes raw HT CSVs for all 5 videos × 40 participants,
extracts features, runs all statistical tests, and saves figures.
"""
import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import shapiro, mannwhitneyu, spearmanr, wilcoxon
from scipy import stats
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import AnovaRM
    HAS_STATSMODELS = True
except ImportError:
    print("WARNING: statsmodels not installed. ANOVA/ANCOVA sections will be skipped.")
    print("  Install with: pip install statsmodels")
    HAS_STATSMODELS = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DATA_DIR = "E:\\Karan\\Sem4\\BRSM\\Project\\data\\data"
HT_DIR = os.path.join(DATA_DIR, "headtracking-data")
OUT_DIR = "E:\\Karan\\Sem4\\BRSM\\Project\\full_analysis_output"
os.makedirs(OUT_DIR, exist_ok=True)

VIDEO_NAMES = {1: "Abandoned Buildings", 2: "Beach Evening", 3: "Campus Walk", 4: "Horror (The Nun)", 5: "Surf"}

# ── 1. Load survey data ──
survey = pd.read_excel(os.path.join(DATA_DIR, "data.xlsx"))
survey['dep_group'] = np.where(survey['score_phq'] >= 10, 'Elevated', 'Minimal/Mild')
print(f"Survey: {len(survey)} participants")
print(f"Depression groups: {survey['dep_group'].value_counts().to_dict()}")

# ── 2. Extract HT features from raw CSVs ──
def extract_features(csv_path):
    """Extract 7 HT features from a single raw CSV."""
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        # Remove first row (often zeros)
        df = df.iloc[1:]
        # Ensure numeric
        for c in ['RotationSpeedTotal','RotationSpeedX','RotationSpeedY','RotationSpeedZ',
                   'RotationChangeX','RotationChangeY','RotationChangeZ']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['RotationSpeedTotal'])
        return {
            'avg_speed_total': df['RotationSpeedTotal'].mean(),
            'avg_speed_x': df['RotationSpeedX'].abs().mean(),
            'avg_speed_y': df['RotationSpeedY'].abs().mean(),
            'avg_speed_z': df['RotationSpeedZ'].abs().mean(),
            'std_rot_x': df['RotationChangeX'].std(),
            'std_rot_y': df['RotationChangeY'].std(),
            'std_rot_z': df['RotationChangeZ'].std(),
        }
    except Exception as e:
        print(f"  WARNING: Error processing {csv_path}: {e}")
        return None

print("\nExtracting HT features for all 5 videos...")
all_features = []
for idx, row in survey.iterrows():
    participant_data = {
        'participant_idx': idx,
        'score_phq': row['score_phq'],
        'score_gad': row['score_gad'],
        'dep_group': row['dep_group'],
        'positive_affect_start': row.get('positive_affect_start', np.nan),
        'negative_affect_start': row.get('negative_affect_start', np.nan),
    }
    for v in range(1, 6):
        ht_file = row[f'v{v}']
        csv_path = os.path.join(HT_DIR, f'v{v}', str(ht_file))
        if os.path.exists(csv_path):
            feats = extract_features(csv_path)
            if feats:
                for k, val in feats.items():
                    participant_data[f'v{v}_{k}'] = val
            else:
                for k in ['avg_speed_total','avg_speed_x','avg_speed_y','avg_speed_z','std_rot_x','std_rot_y','std_rot_z']:
                    participant_data[f'v{v}_{k}'] = np.nan
        else:
            for k in ['avg_speed_total','avg_speed_x','avg_speed_y','avg_speed_z','std_rot_x','std_rot_y','std_rot_z']:
                participant_data[f'v{v}_{k}'] = np.nan
    all_features.append(participant_data)

df = pd.DataFrame(all_features)
print(f"Feature matrix: {df.shape}")

# Check availability
for v in range(1, 6):
    n = df[f'v{v}_avg_speed_total'].notna().sum()
    print(f"  V{v} ({VIDEO_NAMES[v]}): n={n}")

METRICS = ['avg_speed_total', 'avg_speed_x', 'avg_speed_y', 'avg_speed_z', 'std_rot_x', 'std_rot_y', 'std_rot_z']
METRIC_LABELS = {
    'avg_speed_total': 'Total Speed',
    'avg_speed_x': 'Pitch Speed (X)',
    'avg_speed_y': 'Yaw Speed (Y)',
    'avg_speed_z': 'Roll Speed (Z)',
    'std_rot_x': 'Pitch SD (X)',
    'std_rot_y': 'Yaw SD (Y)',
    'std_rot_z': 'Roll SD (Z)',
}

# ── 3. Descriptive Statistics per Video ──
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS PER VIDEO")
print("="*80)
for v in range(1, 6):
    print(f"\n--- V{v}: {VIDEO_NAMES[v]} ---")
    for m in METRICS:
        col = f'v{v}_{m}'
        vals = df[col].dropna()
        print(f"  {METRIC_LABELS[m]:20s}  M={vals.mean():7.2f}  SD={vals.std():7.2f}  Med={vals.median():7.2f}")

# ── 4. Normality Testing (Shapiro-Wilk) per Video ──
print("\n" + "="*80)
print("SHAPIRO-WILK NORMALITY TESTS")
print("="*80)
normality_results = {}
for v in range(1, 6):
    print(f"\n--- V{v}: {VIDEO_NAMES[v]} ---")
    for m in METRICS:
        col = f'v{v}_{m}'
        vals = df[col].dropna()
        if len(vals) >= 8:
            w, p = shapiro(vals)
            normal = "✅" if p > 0.05 else "❌"
            print(f"  {METRIC_LABELS[m]:20s}  W={w:.4f}  p={p:.4f}  {normal}")
            normality_results[(v, m)] = p > 0.05

# ── 5. Mann-Whitney U: Group Comparison per Video ──
print("\n" + "="*80)
print("MANN-WHITNEY U: Minimal/Mild vs Elevated (one-tailed)")
print("Bonferroni α = 0.05/7 = 0.0071")
print("="*80)
mwu_results = []
for v in range(1, 6):
    print(f"\n--- V{v}: {VIDEO_NAMES[v]} ---")
    print(f"  {'Metric':20s} {'Mild M(SD)':>18s} {'Elev M(SD)':>18s} {'U':>6s} {'p':>8s} {'Sig?':>6s}")
    for m in METRICS:
        col = f'v{v}_{m}'
        mild = df[df['dep_group']=='Minimal/Mild'][col].dropna()
        elev = df[df['dep_group']=='Elevated'][col].dropna()
        if len(mild) >= 5 and len(elev) >= 5:
            u, p = mannwhitneyu(mild, elev, alternative='greater')
            sig = "✅" if p < 0.0071 else "✗"
            print(f"  {METRIC_LABELS[m]:20s} {mild.mean():7.2f} ({mild.std():5.2f}) {elev.mean():7.2f} ({elev.std():5.2f}) {u:6.0f} {p:8.4f} {sig:>6s}")
            mwu_results.append({'video': v, 'metric': m, 'mild_mean': mild.mean(), 'elev_mean': elev.mean(), 'U': u, 'p': p})

# ── 6. Spearman Correlation: PHQ-9 vs HT per Video ──
print("\n" + "="*80)
print("SPEARMAN CORRELATION: PHQ-9 (continuous) vs HT metrics")
print("="*80)
for v in range(1, 6):
    print(f"\n--- V{v}: {VIDEO_NAMES[v]} ---")
    print(f"  {'Metric':20s} {'ρ':>8s} {'p':>8s} {'Sig?':>6s}")
    for m in METRICS:
        col = f'v{v}_{m}'
        valid = df[['score_phq', col]].dropna()
        if len(valid) >= 10:
            rho, p = spearmanr(valid['score_phq'], valid[col])
            sig = "✅" if p < 0.05 else ""
            print(f"  {METRIC_LABELS[m]:20s} {rho:8.4f} {p:8.4f} {sig:>6s}")

# ── 7. Wilcoxon Signed-Rank: All Video Pairs ──
print("\n" + "="*80)
print("WILCOXON SIGNED-RANK: Paired Video Comparisons (Total Speed)")
print("="*80)
print(f"  {'Pair':>10s} {'V_a M(SD)':>16s} {'V_b M(SD)':>16s} {'W':>8s} {'p':>10s} {'r':>6s} {'Sig?':>5s}")
for vi in range(1, 6):
    for vj in range(vi+1, 6):
        col_i = f'v{vi}_avg_speed_total'
        col_j = f'v{vj}_avg_speed_total'
        valid = df[[col_i, col_j]].dropna()
        if len(valid) >= 10:
            a = valid[col_i].values
            b = valid[col_j].values
            w, p = wilcoxon(a, b)
            r_eff = abs(stats.norm.ppf(p/2)) / np.sqrt(len(valid))
            sig = "✅" if p < 0.05 else "✗"
            print(f"  V{vi} vs V{vj} {a.mean():7.2f} ({a.std():5.2f}) {b.mean():7.2f} ({b.std():5.2f}) {w:8.1f} {p:10.6f} {r_eff:6.2f} {sig:>5s}")

# ── 8. Cross-Video Mean Profile ──
print("\n" + "="*80)
print("CROSS-VIDEO MEAN TOTAL SPEED BY GROUP")
print("="*80)
print(f"  {'Video':>25s} {'Mild M':>8s} {'Elev M':>8s} {'Diff':>8s}")
for v in range(1, 6):
    col = f'v{v}_avg_speed_total'
    mild_m = df[df['dep_group']=='Minimal/Mild'][col].mean()
    elev_m = df[df['dep_group']=='Elevated'][col].mean()
    print(f"  V{v} {VIDEO_NAMES[v]:>20s} {mild_m:8.2f} {elev_m:8.2f} {mild_m - elev_m:8.2f}")

# ── 9. ANOVA: One-Way (Group) and Repeated-Measures (Video) ──
print("\n" + "="*80)
print("ANOVA ANALYSES")
print("="*80)

# 9a. One-Way ANOVA: Depression Group (Minimal/Mild vs Elevated) per video per metric
print("\n--- 9a. One-Way ANOVA: Depression Group vs HT Metrics (per video) ---")
print("  (Parametric complement to Mann-Whitney; assumes normality)")
print(f"  {'Metric':20s} {'Video':<8} {'F':>8} {'p':>8} {'η²':>8} {'Sig?':>6}")
anova_oneway_results = []
for m in METRICS:
    for v in range(1, 6):
        col = f'v{v}_{m}'
        mild = df[df['dep_group'] == 'Minimal/Mild'][col].dropna()
        elev = df[df['dep_group'] == 'Elevated'][col].dropna()
        if len(mild) >= 5 and len(elev) >= 5:
            F, p = stats.f_oneway(mild, elev)
            # Eta-squared: SS_between / SS_total
            all_vals = np.concatenate([mild.values, elev.values])
            grand_mean = all_vals.mean()
            ss_between = len(mild) * (mild.mean() - grand_mean)**2 + len(elev) * (elev.mean() - grand_mean)**2
            ss_total = np.sum((all_vals - grand_mean)**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
            sig = "✅" if p < 0.05 else ""
            print(f"  {METRIC_LABELS[m]:20s} V{v:<6} {F:8.3f} {p:8.4f} {eta_sq:8.4f} {sig:>6}")
            anova_oneway_results.append({'metric': m, 'video': v, 'F': F, 'p': p, 'eta_sq': eta_sq})

# 9b. One-Way ANOVA: Across Videos (repeated structure collapsed to between) for Total Speed
print("\n--- 9b. One-Way ANOVA: Total Speed across the 5 Videos (N=40 each) ---")
print("  Tests whether video condition significantly affects head movement overall.")
groups_by_video = [df[f'v{v}_avg_speed_total'].dropna().values for v in range(1, 6)]
F_vid, p_vid = stats.f_oneway(*groups_by_video)
all_vid = np.concatenate(groups_by_video)
grand_mean_vid = all_vid.mean()
ss_b = sum(len(g) * (g.mean() - grand_mean_vid)**2 for g in groups_by_video)
ss_t = np.sum((all_vid - grand_mean_vid)**2)
eta_sq_vid = ss_b / ss_t if ss_t > 0 else np.nan
print(f"  F({4}, {len(all_vid)-5}) = {F_vid:.4f},  p = {p_vid:.6f},  η² = {eta_sq_vid:.4f}")
if p_vid < 0.05:
    print("  ✅ Significant: video content significantly affects total head movement speed.")
else:
    print("  ✗ Not significant at α=0.05.")

# 9c. Two-Way ANOVA (video × dep_group) using statsmodels on long-format data
if HAS_STATSMODELS:
    print("\n--- 9c. Two-Way ANOVA: Video × Depression Group (Total Speed) ---")
    print("  Tests main effects of Video, Group, and their interaction.")
    long_rows = []
    for v in range(1, 6):
        col = f'v{v}_avg_speed_total'
        for _, row in df[['dep_group', col]].dropna().iterrows():
            long_rows.append({'video': f'V{v}', 'dep_group': row['dep_group'], 'total_speed': row[col]})
    long_df = pd.DataFrame(long_rows)
    try:
        anova2_model = smf.ols('total_speed ~ C(video) + C(dep_group) + C(video):C(dep_group)',
                               data=long_df).fit()
        anova2_table = sm.stats.anova_lm(anova2_model, typ=2)
        print(anova2_table.to_string())
        # Extract key results
        for src in anova2_table.index:
            row = anova2_table.loc[src]
            pr_val = row.get('PR(>F)', np.nan)
            if not pd.isna(pr_val):
                sig = "✅" if pr_val < 0.05 else "✗"
                print(f"  {src:40s}  p = {pr_val:.4f}  {sig}")
    except Exception as e:
        print(f"  Two-Way ANOVA failed: {e}")
else:
    print("\n--- 9c. Two-Way ANOVA: SKIPPED (statsmodels not available) ---")

# ── 10. ANCOVA: Group Comparison Controlling for Covariates ──
print("\n" + "="*80)
print("ANCOVA ANALYSES")
print("  Tests depression group differences AFTER controlling for confounding variables.")
print("  Covariates: GAD-7 (anxiety), Positive Affect start, Negative Affect start")
print("="*80)

if HAS_STATSMODELS:
    # Identify available covariate columns
    potential_covariates = [
        ('score_gad',            'GAD-7 (Anxiety)'),
        ('positive_affect_start','Positive Affect (PA)'),
        ('negative_affect_start','Negative Affect (NA)'),
    ]
    available_covariates = [(col, lbl) for col, lbl in potential_covariates
                            if col in df.columns and df[col].notna().sum() >= 10]

    if not available_covariates:
        print("  No covariate columns found with sufficient data. ANCOVA skipped.")
    else:
        covar_cols = [c for c, _ in available_covariates]
        covar_lbls = [l for _, l in available_covariates]
        print(f"  Using covariates: {', '.join(covar_lbls)}")

        # Encode depression group as binary (0/1) for regression
        df['dep_binary'] = (df['dep_group'] == 'Elevated').astype(int)

        print(f"\n{'Video':<8} {'Metric':20s} {'Group β':>10} {'Group p':>10} {'F-group':>10} {'η²p':>8} {'Sig?':>6}")
        ancova_results = []
        for v in range(1, 6):
            for m in METRICS:
                col = f'v{v}_{m}'
                # Build working dataframe with complete cases
                work_cols = [col, 'dep_binary'] + covar_cols
                work = df[work_cols].dropna()
                if len(work) < 15:
                    continue  # Not enough data

                # Standardise covariates for interpretable coefficients
                work = work.copy()
                for cc in covar_cols:
                    work[cc] = (work[cc] - work[cc].mean()) / (work[cc].std() + 1e-9)

                # Build OLS formula: outcome ~ dep_group + cov1 + cov2 ...
                cov_formula = ' + '.join(covar_cols)
                formula = f"{col} ~ dep_binary + {cov_formula}"
                formula = formula.replace('-', '_')  # safety for column names
                try:
                    model = smf.ols(formula, data=work).fit()
                    group_coef = model.params.get('dep_binary', np.nan)
                    group_p    = model.pvalues.get('dep_binary', np.nan)

                    # Partial η²: SS_group / (SS_group + SS_residual)
                    # Approximate via t-stat: η²p = t² / (t² + df_resid)
                    t_group = model.tvalues.get('dep_binary', np.nan)
                    df_resid = model.df_resid
                    eta_sq_p = (t_group**2) / (t_group**2 + df_resid) if not np.isnan(t_group) else np.nan
                    F_group = t_group**2  # F = t² for single predictor

                    sig = "✅" if (not np.isnan(group_p) and group_p < 0.05) else ""
                    print(f"  V{v:<5} {METRIC_LABELS[m]:20s} {group_coef:>10.3f} {group_p:>10.4f} "
                          f"{F_group:>10.3f} {eta_sq_p:>8.4f} {sig:>6}")
                    ancova_results.append({
                        'video': v, 'metric': m,
                        'group_beta': group_coef, 'group_p': group_p,
                        'F': F_group, 'eta_sq_partial': eta_sq_p,
                    })
                except Exception as e:
                    print(f"  V{v} {m}: ANCOVA failed — {e}")

        # ANCOVA Summary: significant findings
        sig_ancova = [r for r in ancova_results if r['group_p'] < 0.05]
        print(f"\n  ANCOVA Summary: {len(sig_ancova)}/{len(ancova_results)} metric-video combinations "
              f"show significant group effect after covariate control.")
        if sig_ancova:
            for r in sig_ancova:
                direction = "Elevated > Mild" if r['group_beta'] > 0 else "Mild > Elevated"
                print(f"    V{r['video']} {METRIC_LABELS[r['metric']]}: β={r['group_beta']:.3f}, "
                      f"p={r['group_p']:.4f}, η²p={r['eta_sq_partial']:.4f}  [{direction}]")
else:
    print("  ANCOVA SKIPPED — statsmodels not installed.")
    ancova_results = []

# ── 11. FIGURES (ANOVA & ANCOVA) ──
# Fig F: ANOVA F-statistic heatmap (Group effect per metric × video)
if anova_oneway_results:
    anova_df = pd.DataFrame(anova_oneway_results)
    F_matrix = np.zeros((len(METRICS), 5))
    p_matrix_anova = np.zeros((len(METRICS), 5))
    for _, row in anova_df.iterrows():
        i = METRICS.index(row['metric'])
        j = int(row['video']) - 1
        F_matrix[i, j] = row['F']
        p_matrix_anova[i, j] = row['p']

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(F_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(F_matrix.max(), 1))
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"V{v}\n{VIDEO_NAMES[v]}" for v in range(1, 6)], fontsize=8)
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)
    for i in range(len(METRICS)):
        for j in range(5):
            sig_marker = "*" if p_matrix_anova[i, j] < 0.05 else ""
            ax.text(j, i, f"{F_matrix[i, j]:.2f}{sig_marker}",
                    ha='center', va='center', fontsize=8,
                    color='white' if F_matrix[i, j] > F_matrix.max() * 0.6 else 'black')
    plt.colorbar(im, label="F-statistic")
    ax.set_title("One-Way ANOVA F-Statistics: Depression Group Effect\n(* = p < 0.05; Mild vs Elevated per metric × video)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_f_anova_Fstat_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("\nSaved fig_f_anova_Fstat_heatmap.png")

# Fig G: ANCOVA adjusted group means — total speed per video
if HAS_STATSMODELS and 'ancova_results' in dir() and ancova_results:
    ancova_df_res = pd.DataFrame(ancova_results)
    total_speed_ancova = ancova_df_res[ancova_df_res['metric'] == 'avg_speed_total'].copy()

    if not total_speed_ancova.empty and available_covariates:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: raw group means with ANOVA p-values
        ax = axes[0]
        videos = list(range(1, 6))
        mild_raw = [df[df['dep_group'] == 'Minimal/Mild'][f'v{v}_avg_speed_total'].mean() for v in videos]
        elev_raw = [df[df['dep_group'] == 'Elevated'][f'v{v}_avg_speed_total'].mean() for v in videos]
        x = np.arange(len(videos))
        w = 0.35
        ax.bar(x - w/2, mild_raw, w, label='Minimal/Mild', color='#3B82F6', alpha=0.85)
        ax.bar(x + w/2, elev_raw, w, label='Elevated', color='#EF4444', alpha=0.85)
        # Annotate with one-way ANOVA p
        for j, v in enumerate(videos):
            anova_row = anova_df[(anova_df['video'] == v) & (anova_df['metric'] == 'avg_speed_total')]
            if not anova_row.empty:
                p_val = anova_row.iloc[0]['p']
                ax.text(j, max(mild_raw[j], elev_raw[j]) + 1, f"p={p_val:.3f}",
                        ha='center', fontsize=7, color='red' if p_val < 0.05 else 'grey')
        ax.set_xticks(x)
        ax.set_xticklabels([f"V{v}" for v in videos])
        ax.set_ylabel('Mean Total Speed (deg/s)')
        ax.set_title('One-Way ANOVA: Raw Group Means\n(Total Speed by Depression Group)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Right: ANCOVA group beta coefficients (adjusted effect after covariate control)
        ax2 = axes[1]
        betas = []
        pvals = []
        for v in videos:
            row = total_speed_ancova[total_speed_ancova['video'] == v]
            betas.append(row.iloc[0]['group_beta'] if not row.empty else 0)
            pvals.append(row.iloc[0]['group_p'] if not row.empty else 1)
        colors = ['#EF4444' if p < 0.05 else '#94A3B8' for p in pvals]
        bars = ax2.bar(x, betas, color=colors, alpha=0.85, edgecolor='white')
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        for j, (b, p) in enumerate(zip(betas, pvals)):
            label = f"p={p:.3f}" + (" ✅" if p < 0.05 else "")
            ypos = b + (0.5 if b >= 0 else -1.5)
            ax2.text(j, ypos, label, ha='center', fontsize=7,
                     color='darkred' if p < 0.05 else 'grey')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"V{v}\n{VIDEO_NAMES[v]}" for v in videos], fontsize=8)
        ax2.set_ylabel('Group β (Elevated vs Mild), deg/s')
        ax2.set_title(f'ANCOVA: Adjusted Depression Group Effect\n(controlling for {", ".join(covar_lbls)})')
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle('ANOVA & ANCOVA Comparison: Total Speed by Depression Group', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'fig_g_ancova_adjusted_means.png'), bbox_inches='tight')
        plt.close()
        print("Saved fig_g_ancova_adjusted_means.png")

# ── 12. FIGURES (original) ──
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# Fig A: Cross-video mean speed by group
fig, ax = plt.subplots(figsize=(10, 5))
videos = list(range(1, 6))
mild_means = [df[df['dep_group']=='Minimal/Mild'][f'v{v}_avg_speed_total'].mean() for v in videos]
elev_means = [df[df['dep_group']=='Elevated'][f'v{v}_avg_speed_total'].mean() for v in videos]
x = np.arange(len(videos))
w = 0.35
ax.bar(x - w/2, mild_means, w, label='Minimal/Mild (n=32)', color='#3B82F6', alpha=0.85)
ax.bar(x + w/2, elev_means, w, label='Elevated (n=8)', color='#EF4444', alpha=0.85)
ax.set_xlabel('Video')
ax.set_ylabel('Mean Total Rotation Speed (deg/s)')
ax.set_title('Head Movement by Depression Group Across All 5 Videos')
ax.set_xticks(x)
ax.set_xticklabels([f"V{v}\n{VIDEO_NAMES[v]}" for v in videos], fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_a_crossvideo_groups.png'), bbox_inches='tight')
plt.close()
print("\nSaved fig_a_crossvideo_groups.png")

# Fig B: Boxplots per video per group (Total Speed)
fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
for i, v in enumerate(range(1, 6)):
    ax = axes[i]
    col = f'v{v}_avg_speed_total'
    mild = df[df['dep_group']=='Minimal/Mild'][col].dropna()
    elev = df[df['dep_group']=='Elevated'][col].dropna()
    bp = ax.boxplot([mild, elev], labels=['Mild', 'Elev'], patch_artist=True,
                    boxprops=dict(alpha=0.7), widths=0.6)
    bp['boxes'][0].set_facecolor('#3B82F6')
    bp['boxes'][1].set_facecolor('#EF4444')
    # Add individual points
    for j, (data, xpos) in enumerate([(mild, 1), (elev, 2)]):
        ax.scatter(np.random.normal(xpos, 0.05, len(data)), data, alpha=0.4, s=20, color='black', zorder=3)
    u, p = mannwhitneyu(mild, elev, alternative='greater')
    ax.set_title(f"V{v}: {VIDEO_NAMES[v]}\np={p:.4f}", fontsize=9)
    if i == 0:
        ax.set_ylabel('Total Speed (deg/s)')
plt.suptitle('Group Comparison: Total Speed Across All 5 Videos (Mann-Whitney U)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_b_boxplots_all_videos.png'), bbox_inches='tight')
plt.close()
print("Saved fig_b_boxplots_all_videos.png")

# Fig C: Heatmap of Spearman correlations (PHQ-9 vs all metrics × all videos)
corr_matrix = np.zeros((len(METRICS), 5))
pval_matrix = np.zeros((len(METRICS), 5))
for j, v in enumerate(range(1, 6)):
    for i, m in enumerate(METRICS):
        col = f'v{v}_{m}'
        valid = df[['score_phq', col]].dropna()
        if len(valid) >= 10:
            rho, p = spearmanr(valid['score_phq'], valid[col])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = p

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-0.4, vmax=0.4, aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels([f"V{v}\n{VIDEO_NAMES[v]}" for v in range(1,6)], fontsize=8)
ax.set_yticks(range(len(METRICS)))
ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)
# Add text annotations
for i in range(len(METRICS)):
    for j in range(5):
        sig = "*" if pval_matrix[i,j] < 0.05 else ""
        ax.text(j, i, f"{corr_matrix[i,j]:.2f}{sig}", ha='center', va='center', fontsize=8,
                color='white' if abs(corr_matrix[i,j]) > 0.25 else 'black')
plt.colorbar(im, label="Spearman ρ")
ax.set_title("PHQ-9 vs Head-Tracking: Spearman Correlations (* = p<0.05)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_c_correlation_heatmap.png'), bbox_inches='tight')
plt.close()
print("Saved fig_c_correlation_heatmap.png")

# Fig D: Video profile — mean total speed across videos
fig, ax = plt.subplots(figsize=(10, 5))
video_means = [df[f'v{v}_avg_speed_total'].mean() for v in range(1,6)]
video_sds = [df[f'v{v}_avg_speed_total'].std() for v in range(1,6)]
bars = ax.bar(range(5), video_means, yerr=video_sds, capsize=5,
              color=['#F59E0B', '#3B82F6', '#10B981', '#EF4444', '#8B5CF6'], alpha=0.85, edgecolor='white')
ax.set_xticks(range(5))
ax.set_xticklabels([f"V{v}\n{VIDEO_NAMES[v]}" for v in range(1,6)], fontsize=9)
ax.set_ylabel('Mean Total Speed (deg/s)')
ax.set_title('Mean Head Movement Speed Across All 5 Videos (N=40)')
ax.grid(axis='y', alpha=0.3)
# Add mean values on bars
for bar, mean in zip(bars, video_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{mean:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_d_video_profile.png'), bbox_inches='tight')
plt.close()
print("Saved fig_d_video_profile.png")

# Fig E: Per-video Mann-Whitney p-values heatmap
pval_mwu = np.zeros((len(METRICS), 5))
for j, v in enumerate(range(1, 6)):
    for i, m in enumerate(METRICS):
        col = f'v{v}_{m}'
        mild = df[df['dep_group']=='Minimal/Mild'][col].dropna()
        elev = df[df['dep_group']=='Elevated'][col].dropna()
        if len(mild) >= 5 and len(elev) >= 5:
            u, p = mannwhitneyu(mild, elev, alternative='greater')
            pval_mwu[i, j] = p

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pval_mwu, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels([f"V{v}\n{VIDEO_NAMES[v]}" for v in range(1,6)], fontsize=8)
ax.set_yticks(range(len(METRICS)))
ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)
for i in range(len(METRICS)):
    for j in range(5):
        sig = "★" if pval_mwu[i,j] < 0.0071 else ""
        ax.text(j, i, f"{pval_mwu[i,j]:.3f}{sig}", ha='center', va='center', fontsize=8,
                color='white' if pval_mwu[i,j] < 0.2 else 'black')
plt.colorbar(im, label="p-value (one-tailed)")
ax.set_title("Mann-Whitney U p-values: Mild > Elevated\n(★ = p < 0.0071 Bonferroni)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_e_pvalue_heatmap.png'), bbox_inches='tight')
plt.close()
print("Saved fig_e_pvalue_heatmap.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE — All results saved to:", OUT_DIR)
print("="*80)
