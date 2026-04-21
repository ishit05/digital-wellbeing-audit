import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# HELPER: parse Samsung/StayFree time strings → minutes (float)
#   e.g. "1h 10m 45s" → 70.75,  "2m 30s" → 2.5,  "0s" → 0.0
# ─────────────────────────────────────────────────────────────────
def parse_time(val):
    if pd.isna(val) or str(val).strip() in ['', 'nan', '0s', '0']:
        return 0.0
    s = str(val).strip()
    h = re.search(r'(\d+)h', s)
    m = re.search(r'(\d+)m', s)
    sec = re.search(r'(\d+)s', s)
    total = 0.0
    if h:  total += int(h.group(1)) * 60
    if m:  total += int(m.group(1))
    if sec: total += int(sec.group(1)) / 60
    return round(total, 2)

# ─────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN SCREEN TIME
#    Format: rows = apps, cols = [App, Device, date1, date2, Total]
# ─────────────────────────────────────────────────────────────────
st_raw = pd.read_csv("screentime.csv")

# Drop footer rows (NaN app name or "Created by", "Creation date")
st_raw = st_raw.dropna(subset=[st_raw.columns[0]])
st_raw = st_raw[~st_raw[st_raw.columns[0]].str.startswith('Created', na=False)]
st_raw = st_raw[~st_raw[st_raw.columns[0]].str.startswith('Creation', na=False)]
st_raw = st_raw[st_raw[st_raw.columns[0]] != 'Total Usage']   # remove summary row

# Rename columns
st_raw.columns = ['App', 'Device', 'Apr_20', 'Apr_21', 'Total']

# Parse time columns to minutes
for col in ['Apr_20', 'Apr_21', 'Total']:
    st_raw[col] = st_raw[col].apply(parse_time)

st_raw = st_raw.reset_index(drop=True)
print("=== Screen Time Data (minutes) ===")
print(st_raw[['App', 'Apr_20', 'Apr_21', 'Total']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────
# 2. APP CATEGORY MAPPING
# ─────────────────────────────────────────────────────────────────
social_apps   = ['Instagram', 'Snapchat', 'WhatsApp', 'YouTube', 'Facebook', 'Twitter', 'TikTok']
utility_apps  = ['Chrome', 'Drive', 'Files', 'Gallery', 'Settings', 'Camera',
                 'Google Play Store', 'Google Play services', 'Wallet',
                 'Digital Wellbeing', 'StayFree', 'One UI Home', 'Permission controller',
                 'Settings Suggestions', 'Photo Screensavers', 'Photos & videos',
                 'Android System', 'Navi']
payment_apps  = ['Paytm', 'PhonePe']
comm_apps     = ['Call', 'Phone', 'Messages', 'Truecaller']
prod_apps     = ['ChatGPT', 'Amazon']

def categorise(app):
    if app in social_apps:   return 'Social Media'
    if app in comm_apps:     return 'Communication'
    if app in payment_apps:  return 'Payments'
    if app in prod_apps:     return 'Productivity'
    if app in utility_apps:  return 'System/Utility'
    return 'Other'

st_raw['Category'] = st_raw['App'].apply(categorise)

# Daily totals
total_apr20 = st_raw['Apr_20'].sum()
total_apr21 = st_raw['Apr_21'].sum()
print(f"\nTotal screen time  Apr 20: {total_apr20:.1f} min ({total_apr20/60:.2f} hrs)")
print(f"Total screen time  Apr 21: {total_apr21:.1f} min ({total_apr21/60:.2f} hrs)")

# Per-category breakdown
cat_20 = st_raw.groupby('Category')['Apr_20'].sum().sort_values(ascending=False)
cat_21 = st_raw.groupby('Category')['Apr_21'].sum().sort_values(ascending=False)

print("\n=== Category Breakdown (Apr 20) ===")
print(cat_20.to_string())
print("\n=== Category Breakdown (Apr 21) ===")
print(cat_21.to_string())

# ─────────────────────────────────────────────────────────────────
# 3. LOAD MOOD / PRODUCTIVITY / SLEEP (Survey)
# ─────────────────────────────────────────────────────────────────
import os, glob

# Auto-detect the mood CSV regardless of exact filename
_candidates = glob.glob("*Form responses*") + glob.glob("*Digital Wellbeing*") + \
              glob.glob("*_Form_responses*") + glob.glob("*Wellbeing*Audit*")
if not _candidates:
    raise FileNotFoundError(
        "Could not find the mood/form CSV. Make sure it is in the same folder as this script.\n"
        f"Files found: {os.listdir('.')}"
    )
_mood_file = _candidates[0]
print(f"Loading mood data from: {_mood_file}")
mood_raw = pd.read_csv(_mood_file)
# Rename columns for convenience
mood_raw.columns = ['Timestamp', 'Mood', 'Productivity', 'Sleep_hrs']
mood_raw['Timestamp'] = pd.to_datetime(mood_raw['Timestamp'], dayfirst=True)
mood_raw['Date'] = mood_raw['Timestamp'].dt.date

print("\n=== Mood / Productivity Survey (all responses) ===")
print(mood_raw[['Timestamp', 'Mood', 'Productivity', 'Sleep_hrs']].to_string(index=False))

# Summary stats
print("\n=== Survey Summary Statistics ===")
print(mood_raw[['Mood', 'Productivity', 'Sleep_hrs']].describe().round(2))

avg_mood  = mood_raw['Mood'].mean()
avg_prod  = mood_raw['Productivity'].mean()
avg_sleep = mood_raw['Sleep_hrs'].mean()
print(f"\nAverage Mood:         {avg_mood:.2f} / 5")
print(f"Average Productivity: {avg_prod:.2f} / 5")
print(f"Average Sleep:        {avg_sleep:.2f} hrs")

# ─────────────────────────────────────────────────────────────────
# 4. CORRELATIONS (within survey responses)
# ─────────────────────────────────────────────────────────────────
corr_sleep_mood  = mood_raw['Sleep_hrs'].corr(mood_raw['Mood'])
corr_sleep_prod  = mood_raw['Sleep_hrs'].corr(mood_raw['Productivity'])
corr_mood_prod   = mood_raw['Mood'].corr(mood_raw['Productivity'])

print("\n=== Correlations ===")
print(f"Sleep  ↔ Mood:         {corr_sleep_mood:+.2f}")
print(f"Sleep  ↔ Productivity: {corr_sleep_prod:+.2f}")
print(f"Mood   ↔ Productivity: {corr_mood_prod:+.2f}")

# Social media time (Apr 20 – main full day)
social_total_20 = st_raw[st_raw['Category'] == 'Social Media']['Apr_20'].sum()
social_total_21 = st_raw[st_raw['Category'] == 'Social Media']['Apr_21'].sum()
print(f"\nSocial Media Time  Apr 20: {social_total_20:.1f} min")
print(f"Social Media Time  Apr 21: {social_total_21:.1f} min")

# ─────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────
palette = {
    'Social Media':    '#e74c3c',
    'Communication':   '#3498db',
    'System/Utility':  '#95a5a6',
    'Productivity':    '#2ecc71',
    'Payments':        '#f39c12',
    'Other':           '#9b59b6',
}
sns.set_theme(style='whitegrid', palette='muted')
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Digital Wellbeing Audit – Screen Time vs Mood & Productivity',
             fontsize=16, fontweight='bold', y=0.98)

# ── 5.1  App usage bar chart (Apr 20, top 12 apps) ──────────────
ax1 = fig.add_subplot(3, 3, 1)
top_apps = st_raw[st_raw['Apr_20'] > 0].nlargest(12, 'Apr_20')
colors_bar = [palette.get(c, '#bdc3c7') for c in top_apps['Category']]
bars = ax1.barh(top_apps['App'], top_apps['Apr_20'], color=colors_bar, edgecolor='white')
ax1.set_xlabel('Minutes')
ax1.set_title('Top Apps – Apr 20 (minutes)', fontweight='bold')
ax1.invert_yaxis()
for bar, val in zip(bars, top_apps['Apr_20']):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.0f}m', va='center', fontsize=7)
patches = [mpatches.Patch(color=v, label=k) for k, v in palette.items()]
ax1.legend(handles=patches, fontsize=6, loc='lower right')

# ── 5.2  Category pie – Apr 20 ──────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
cat_data = st_raw[st_raw['Apr_20'] > 0].groupby('Category')['Apr_20'].sum()
cat_colors = [palette.get(c, '#bdc3c7') for c in cat_data.index]
wedges, texts, autotexts = ax2.pie(
    cat_data, labels=cat_data.index, autopct='%1.1f%%',
    colors=cat_colors, startangle=140,
    textprops={'fontsize': 8}
)
ax2.set_title('Category Split – Apr 20', fontweight='bold')

# ── 5.3  Day comparison bar ──────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 3)
days   = ['Apr 20', 'Apr 21']
totals = [total_apr20, total_apr21]
bar_colors = ['#3498db', '#e67e22']
b = ax3.bar(days, totals, color=bar_colors, width=0.5, edgecolor='white')
ax3.set_ylabel('Minutes')
ax3.set_title('Total Screen Time by Day', fontweight='bold')
for bar, val in zip(b, totals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val:.0f}m\n({val/60:.1f}h)', ha='center', fontsize=9)
ax3.set_ylim(0, max(totals) * 1.25)

# ── 5.4  Mood distribution ──────────────────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
mood_counts = mood_raw['Mood'].value_counts().sort_index()
ax4.bar(mood_counts.index, mood_counts.values, color='#e74c3c',
        edgecolor='white', width=0.6)
ax4.axvline(avg_mood, color='black', linestyle='--', linewidth=1.5,
            label=f'Mean = {avg_mood:.1f}')
ax4.set_xlabel('Mood Score (1=Very Bad, 5=Very Good)')
ax4.set_ylabel('Count')
ax4.set_title('Mood Distribution', fontweight='bold')
ax4.set_xticks([1, 2, 3, 4, 5])
ax4.legend(fontsize=9)

# ── 5.5  Productivity distribution ──────────────────────────────
ax5 = fig.add_subplot(3, 3, 5)
prod_counts = mood_raw['Productivity'].value_counts().sort_index()
ax5.bar(prod_counts.index, prod_counts.values, color='#2ecc71',
        edgecolor='white', width=0.6)
ax5.axvline(avg_prod, color='black', linestyle='--', linewidth=1.5,
            label=f'Mean = {avg_prod:.1f}')
ax5.set_xlabel('Productivity Score (1=None, 5=Very High)')
ax5.set_ylabel('Count')
ax5.set_title('Productivity Distribution', fontweight='bold')
ax5.set_xticks([1, 2, 3, 4, 5])
ax5.legend(fontsize=9)

# ── 5.6  Sleep distribution ─────────────────────────────────────
ax6 = fig.add_subplot(3, 3, 6)
ax6.hist(mood_raw['Sleep_hrs'], bins=range(2, 14), color='#9b59b6',
         edgecolor='white', align='left')
ax6.axvline(avg_sleep, color='black', linestyle='--', linewidth=1.5,
            label=f'Mean = {avg_sleep:.1f}h')
ax6.set_xlabel('Sleep Hours')
ax6.set_ylabel('Count')
ax6.set_title('Sleep Hours Distribution', fontweight='bold')
ax6.legend(fontsize=9)

# ── 5.7  Mood vs Productivity scatter ───────────────────────────
ax7 = fig.add_subplot(3, 3, 7)
sns.regplot(data=mood_raw, x='Mood', y='Productivity', ax=ax7,
            scatter_kws={'alpha': 0.7, 's': 60, 'color': '#3498db'},
            line_kws={'color': 'red'})
ax7.set_title(f'Mood vs Productivity\n(r = {corr_mood_prod:+.2f})', fontweight='bold')
ax7.set_xlim(0.5, 5.5); ax7.set_ylim(0.5, 5.5)
ax7.set_xticks([1,2,3,4,5]); ax7.set_yticks([1,2,3,4,5])

# ── 5.8  Sleep vs Mood scatter ──────────────────────────────────
ax8 = fig.add_subplot(3, 3, 8)
sns.regplot(data=mood_raw, x='Sleep_hrs', y='Mood', ax=ax8,
            scatter_kws={'alpha': 0.7, 's': 60, 'color': '#e74c3c'},
            line_kws={'color': 'navy'})
ax8.set_title(f'Sleep vs Mood\n(r = {corr_sleep_mood:+.2f})', fontweight='bold')

# ── 5.9  Sleep vs Productivity scatter ──────────────────────────
ax9 = fig.add_subplot(3, 3, 9)
sns.regplot(data=mood_raw, x='Sleep_hrs', y='Productivity', ax=ax9,
            scatter_kws={'alpha': 0.7, 's': 60, 'color': '#2ecc71'},
            line_kws={'color': 'navy'})
ax9.set_title(f'Sleep vs Productivity\n(r = {corr_sleep_prod:+.2f})', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('wellbeing_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Plot saved: wellbeing_plots.png")

# ─────────────────────────────────────────────────────────────────
# 6. ACTIONABLE INSIGHTS
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("            ACTIONABLE INSIGHTS")
print("="*55)

# Social media
social_pct = (social_total_20 / total_apr20 * 100) if total_apr20 > 0 else 0
print(f"\n📱 Social Media Usage (Apr 20): {social_total_20:.0f} min ({social_pct:.1f}% of day)")
if social_total_20 > 90:
    cut = social_total_20 * 0.25
    print(f"   ⚠  That's over 1.5 hrs. Reducing by 25% (~{cut:.0f} min) frees up "
          f"meaningful time.")
elif social_total_20 > 60:
    print("   🟡 Moderate usage. Consider a 30-min daily cap.")
else:
    print("   ✅ Social media usage looks healthy.")

# YouTube
yt_row = st_raw[st_raw['App'] == 'YouTube']
if not yt_row.empty:
    yt_time = yt_row['Apr_20'].values[0]
    print(f"\n📺 YouTube: {yt_time:.0f} min on Apr 20 – that's "
          f"{yt_time/60:.1f} hrs.")
    if yt_time > 60:
        print("   ⚠  Consider batching YouTube into scheduled slots.")

# Sleep-mood insight
print(f"\n😴 Sleep ↔ Mood correlation: {corr_sleep_mood:+.2f}")
if corr_sleep_mood > 0.4:
    print("   ✅ More sleep is associated with better mood. Protect 7-9 hrs.")
elif corr_sleep_mood < -0.3:
    print("   ⚠  Negative correlation detected – check for confounding factors.")
else:
    print("   ℹ  No strong sleep-mood pattern in current data.")

# Mood-productivity insight
print(f"\n😊 Mood ↔ Productivity correlation: {corr_mood_prod:+.2f}")
if corr_mood_prod > 0.4:
    print("   ✅ Higher mood links to higher productivity. Invest in wellbeing.")
elif corr_mood_prod < -0.3:
    print("   ⚠  Low mood yet high productivity – watch out for burnout.")
else:
    print("   ℹ  Mood and productivity appear independent in this sample.")

print(f"\n📊 Average wellbeing scores:")
print(f"   Mood:         {avg_mood:.1f} / 5  {'😊' if avg_mood >= 3.5 else '😐' if avg_mood >= 2.5 else '😞'}")
print(f"   Productivity: {avg_prod:.1f} / 5  {'🚀' if avg_prod >= 3.5 else '🟡' if avg_prod >= 2.5 else '🔴'}")
print(f"   Sleep:        {avg_sleep:.1f} hrs  {'✅' if avg_sleep >= 7 else '⚠  Below recommended 7 hrs'}")

print("\n" + "="*55)
print("Analysis complete. Output saved to wellbeing_plots.png")