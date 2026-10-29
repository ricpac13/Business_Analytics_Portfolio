#!/usr/bin/env python
# coding: utf-8

# # Impact of 2018 Promotion on Loyalty Memberships and Retention
# ### Richard Pacifico 
# ### Business Analytics Fall 2025

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

results_path = 'Results.csv'
flights_path = 'flight activity.csv'

results = pd.read_csv(results_path)
flights = pd.read_csv(flights_path)


# In[2]:


cust = results.copy()
cust['Enroll_Date'] = pd.to_datetime(
    cust['Enrollment_Year'].astype(int).astype(str) + '-' +
    cust['Enrollment_Month'].astype(int).astype(str),
    errors='coerce'
)

cancel_dates = pd.to_datetime(
    cust['Cancellation_Year'].astype('Int64').astype(str) + '-' +
    cust['Cancellation_Month'].astype('Int64').astype(str),
    errors='coerce'
)
cust['Cancel_Date'] = cancel_dates
cust['Enrollment_Quarter'] = cust['Enroll_Date'].dt.to_period('Q')
cust['Cancellation_Quarter'] = cust['Cancel_Date'].dt.to_period('Q')
cust['Promo_Flag'] = cust['Enrollment_Type'].eq('2018 Promotion')
cust['Canceled'] = cust['Cancellation_Year'].notna()


# In[3]:


cust[['Loyalty_Number', 'Enrollment_Type', 'Promo_Flag', 'Enrollment_Quarter']].head()


# In[4]:


#Figure out what quarter a promotion was available + the range of the data/last available quarter
promo_quarters = (
    cust.loc[cust['Promo_Flag'], 'Enrollment_Quarter']
        .dropna()
        .sort_values()
        .unique()
)

last_quarter = cust['Enrollment_Quarter'].max()

print('quarters with promotion available:', promo_quarters)
print('Last enrollment quarter:', last_quarter)


# In[5]:


cohort = cust[cust['Enrollment_Quarter'].isin(promo_quarters)].copy()
print('Total customers enrolled durnig promotion quarters:', cohort['Loyalty_Number'].nunique())


# # Retention and Cancellation Timing Relative to Last Quarter
# 

# In[6]:


#Retention to last_quarter
cohort['Retained_to_last'] = ~(
    cohort['Canceled'] & (cohort['Cancellation_Quarter'] < last_quarter)
)

def cancel_category(row):
    if pd.isna(row['Cancellation_Quarter']):
        return 'Retained_through_last'
    if row['Cancellation_Quarter'] < last_quarter:
        return 'Canceled_before_last'
    if row['Cancellation_Quarter'] == last_quarter:
        return 'Canceled_in_last'
    if row['Cancellation_Quarter'] > last_quarter:
        return 'Retained_through_last'
    return 'Retained_through_last'

cohort['Cancel_Timing'] = cohort.apply(cancel_category, axis=1)
#cohort[['Loyalty_Number', 'Promo_Flag', 'Enrollment_Quarter', 'Cancellation_Quarter', 'Retained_to_last', 'Cancel_Timing']].head()


# In[7]:


#Gross acquisitions by promo vs standard (during promo quarters)
acq_summary = (
    cohort.groupby('Promo_Flag')
          .agg(acquisitions=('Loyalty_Number', 'nunique'))
)
acq_summary.index = acq_summary.index.map({False: 'Standard', True: 'Promotion'})
acq_summary


# In[8]:


ret_summary = (
    cohort.groupby('Promo_Flag')
          .agg(
              customers=('Loyalty_Number', 'nunique'),
              retained_to_last=('Retained_to_last', 'sum')
          )
)
ret_summary['retention_rate_to_last'] = ret_summary['retained_to_last'] / ret_summary['customers']
ret_summary.index = ret_summary.index.map({False: 'Standard', True: 'Promotion'})
ret_summary


# In[9]:


cancel_timing_summary = (
    cohort.groupby(['Promo_Flag', 'Cancel_Timing'])
          .agg(customers=('Loyalty_Number', 'nunique'))
          .reset_index()
)
cancel_timing_summary['Promo_Label'] = cancel_timing_summary['Promo_Flag'].map({False: 'Standard', True: 'Promotion'})
cancel_timing_summary


# In[10]:


fig, ax = plt.subplots(figsize=(8, 5))

pivot_ct = cancel_timing_summary.pivot(
    index="Promo_Label",
    columns="Cancel_Timing",
    values="customers"
).fillna(0)

#Convert to percentages
row_sums = pivot_ct.sum(axis=1)
pivot_pct = pivot_ct.div(row_sums, axis=0)

bottom = np.zeros(len(pivot_pct.index))
for col in ["Canceled_before_last", "Canceled_in_last", "Retained_through_last"]:
    if col in pivot_pct.columns:
        vals = pivot_pct[col].values
        bars = ax.bar(pivot_pct.index.astype(str), vals, bottom=bottom, label=col)
        bottom += vals

ax.set_ylabel("Share of customers")
ax.set_title("Cancellation Timing (% of customers) by Type")
ax.legend(title="Cancel timing")

plt.tight_layout()
plt.show()


# In[11]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Gross acquisitions
axes[0].bar(acq_summary.index.astype(str), acq_summary['acquisitions'])
axes[0].set_title('Acquisitions During Promotion Quarters')
axes[0].set_ylabel('Number of customers')

# Retention rate to last_quarter
axes[1].bar(ret_summary.index.astype(str), ret_summary['retention_rate_to_last'])
axes[1].set_title('Administrative Retention to Last Quarter')
axes[1].set_ylabel('Retention rate')

plt.tight_layout()
plt.show()


# In[12]:


# Merge cohort customers with flight activity
activity = pd.merge(
    flights,
    cohort[['Loyalty_Number', 'Promo_Flag']],
    on='Loyalty_Number',
    how='inner'
)

#Build Flight_Date and Quarter
activity['Flight_Date'] = pd.to_datetime(
    activity['Year'].astype(int).astype(str) + '-' +
    activity['Month'].astype(int).astype(str),
    errors='coerce'
)
activity['Flight_Quarter'] = activity['Flight_Date'].dt.to_period('Q')

# Filter to flights in the last quarter only
activity_last_q = activity[activity['Flight_Quarter'] == last_quarter].copy()
activity_last_q.head()


# In[13]:


# Flag customers with any flights in last quarter
active_last = (
    activity_last_q.groupby('Loyalty_Number')['Total_Flights']
                   .sum()
                   .rename('Flights_in_last_q')
)

# Join back to cohort
cohort = cohort.merge(active_last, on='Loyalty_Number', how='left')
cohort['Flights_in_last_q'] = cohort['Flights_in_last_q'].fillna(0)
cohort['Active_in_last_q'] = cohort['Flights_in_last_q'] > 0

cohort[['Loyalty_Number', 'Promo_Flag', 'Retained_to_last', 'Flights_in_last_q', 'Active_in_last_q']].head()


# In[14]:


# Compare activity rates in last quarter by promo vs standard
activity_summary = (
    cohort.groupby('Promo_Flag')
          .agg(
              customers=('Loyalty_Number', 'nunique'),
              active_in_last_q=('Active_in_last_q', 'mean'),
              avg_flights_last_q=('Flights_in_last_q', 'mean')
          )
)
activity_summary.index = activity_summary.index.map({False: 'Standard', True: 'Promotion'})
activity_summary


# In[15]:


fig, ax = plt.subplots(1, 1, figsize=(5, 4))

ax.bar(activity_summary.index.astype(str), activity_summary['avg_flights_last_q'])
ax.set_title('Average Flights in Last Quarter')
ax.set_ylabel('Average flights')

plt.tight_layout()
plt.show()


# In[16]:


# Create salary bands
cohort['Salary_Band'] = pd.cut(
    cohort['Salary'],
    bins=[0, 50000, 100000, 150000, 300000, float('inf')],
    labels=['0–50k', '50–100k', '100–150k', '150–300k', '300k+'],
    include_lowest=True
)

# CLV quartiles
cohort['CLV_Quartile'] = pd.qcut(cohort['CLV'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

cohort[['Loyalty_Number', 'Promo_Flag', 'Salary_Band', 'CLV_Quartile', 'Gender', 'Education', 'Loyalty_Card']].head()


# In[17]:


def segment_summary(df, segment_col):
    """Return retention and activity summary by segment and promo flag."""
    summary = (
        df.groupby([segment_col, 'Promo_Flag'])
          .agg(
              customers=('Loyalty_Number', 'nunique'),
              retention_to_last=('Retained_to_last', 'mean'),
              active_in_last_q=('Active_in_last_q', 'mean')
          )
          .reset_index()
    )
    summary['Promo_Label'] = summary['Promo_Flag'].map({False: 'Standard', True: 'Promotion'})
    return summary

seg_salary = segment_summary(cohort, 'Salary_Band')
seg_gender = segment_summary(cohort, 'Gender')
seg_education = segment_summary(cohort, 'Education')
seg_clv = segment_summary(cohort, 'CLV_Quartile')
seg_tier = segment_summary(cohort, 'Loyalty_Card')

seg_salary


# In[18]:


pivot_salary = seg_salary.pivot(
    index='Salary_Band',
    columns='Promo_Label',
    values='retention_to_last'
)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(pivot_salary.index))
width = 0.35

standard_vals = pivot_salary['Standard'] if 'Standard' in pivot_salary.columns else np.zeros(len(x))
promo_vals = pivot_salary['Promotion'] if 'Promotion' in pivot_salary.columns else np.zeros(len(x))

ax.bar(x - width/2, standard_vals, width, label='Standard')
ax.bar(x + width/2, promo_vals, width, label='Promotion')

ax.set_xticks(x)
ax.set_xticklabels(pivot_salary.index.astype(str), rotation=45)
ax.set_ylabel('Retention rate to last quarter')
ax.set_title('Retention by Salary Band and Promotion Flag')
ax.legend()
plt.tight_layout()
plt.show()


# In[19]:


acq_summary



# In[20]:


ret_summary


# In[ ]:





# In[21]:


activity_summary


# In[ ]:





# In[22]:


seg_salary


# In[ ]:





# In[ ]:





# In[23]:


seg_tier


# In[24]:


# Make labels show nicely if available
def safe_bar_label(bars, fmt="%.1f"):
    try:
        from matplotlib.pyplot import bar_label  # matplotlib >=3.4
        bar_label(bars, fmt=fmt)
    except Exception:
        # Older matplotlib, just skip labels
        pass

# 1) Overview: Acquisitions vs Retained-to-Last
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(ret_summary.index))  # Standard, Promotion
width = 0.35

acq_vals = acq_summary.loc[ret_summary.index, "acquisitions"].values
ret_vals = ret_summary["retained_to_last"].values

bars1 = ax.bar(x - width/2, acq_vals, width, label="Acquisitions")
bars2 = ax.bar(x + width/2, ret_vals, width, label="Retained to last quarter")

ax.set_xticks(x)
ax.set_xticklabels(ret_summary.index.astype(str))
ax.set_ylabel("Number of customers")
ax.set_title("Acquisitions vs Retained-to-Last (Promo vs Standard)")
ax.legend()

safe_bar_label(bars1, fmt="%.0f")
safe_bar_label(bars2, fmt="%.0f")

plt.tight_layout()
plt.show()


# In[25]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Pivot retention and activity by salary band
pivot_sal_ret = seg_salary.pivot(
    index="Salary_Band",
    columns="Promo_Label",
    values="retention_to_last"
)
pivot_sal_act = seg_salary.pivot(
    index="Salary_Band",
    columns="Promo_Label",
    values="active_in_last_q"
)

bands = pivot_sal_ret.index.astype(str)
x = np.arange(len(bands))

# Left: Retention to last quarter
width = 0.35
std_ret = pivot_sal_ret["Standard"] if "Standard" in pivot_sal_ret.columns else np.zeros(len(x))
promo_ret = pivot_sal_ret["Promotion"] if "Promotion" in pivot_sal_ret.columns else np.zeros(len(x))

b1 = axes[0].bar(x - width/2, std_ret, width, label="Standard")
b2 = axes[0].bar(x + width/2, promo_ret, width, label="Promotion")

axes[0].set_xticks(x)
axes[0].set_xticklabels(bands, rotation=45)
axes[0].set_ylabel("Retention rate")
axes[0].set_title("Retention to Last Quarter by Salary Band")
axes[0].legend()

# Right: Active in last quarter
std_act = pivot_sal_act["Standard"] if "Standard" in pivot_sal_act.columns else np.zeros(len(x))
promo_act = pivot_sal_act["Promotion"] if "Promotion" in pivot_sal_act.columns else np.zeros(len(x))

axes[1].bar(x - width/2, std_act, width, label="Standard")
axes[1].bar(x + width/2, promo_act, width, label="Promotion")

axes[1].set_xticks(x)
axes[1].set_xticklabels(bands, rotation=45)
axes[1].set_ylabel("Active rate in last quarter")
axes[1].set_title("Q4 Activity by Salary Band")
axes[1].legend()

plt.tight_layout()
plt.show()


# In[26]:


enroll_by_q = (
    cust.groupby(["Enrollment_Quarter", "Promo_Flag"])
        .agg(enrollments=("Loyalty_Number", "nunique"))
        .reset_index()
)

enroll_by_q["Promo_Label"] = enroll_by_q["Promo_Flag"].map({True: "Promotion", False: "Standard"})

enroll_pivot = enroll_by_q.pivot(
    index="Enrollment_Quarter",
    columns="Promo_Label",
    values="enrollments"
).fillna(0)

quarters = enroll_pivot.index.astype(str)
x = np.arange(len(quarters))
width = 0.35

std_vals = enroll_pivot["Standard"] if "Standard" in enroll_pivot.columns else np.zeros(len(x))
promo_vals = enroll_pivot["Promotion"] if "Promotion" in enroll_pivot.columns else np.zeros(len(x))

fig, ax = plt.subplots(figsize=(10,5))
b1 = ax.bar(x - width/2, std_vals, width, label="Standard")
b2 = ax.bar(x + width/2, promo_vals, width, label="Promotion")

ax.set_xticks(x)
ax.set_xticklabels(quarters, rotation=45)
ax.set_ylabel("Enrollments")
ax.set_title("Enrollments by Quarter: Promotion vs Standard")
ax.legend()
ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()


# In[38]:


cust = cohort[['Loyalty_Number','Promo_Flag','Enroll_Date','Salary_Band']].drop_duplicates('Loyalty_Number').copy()
cust['Enroll_Month_dt'] = pd.to_datetime(cust['Enroll_Date']).dt.to_period('M').dt.to_timestamp()

window_start = pd.Timestamp("2018-01-01")
window_end   = pd.Timestamp("2018-12-31")
all_months   = pd.date_range(window_start, window_end, freq="MS")
cust = cust[cust['Enroll_Month_dt'] <= window_end]

fa = flights.copy()
date_col = next((c for c in ['Flight_Date','FlightDate','Date','date','Activity_Date','Transaction_Date'] if c in fa.columns), None)

if date_col:
    fa['Flight_Month_dt'] = pd.to_datetime(fa[date_col], errors='coerce').dt.to_period('M').dt.to_timestamp()
else:
    mcol = next((c for c in ['Month','month','Flight_Month','Activity_Month'] if c in fa.columns), None)
    ycol = next((c for c in ['Year','year','Flight_Year','Activity_Year'] if c in fa.columns), None)
    fa['Flight_Month_dt'] = pd.to_datetime(fa[ycol].astype(str) + "-" + fa[mcol].astype(str) + "-01", errors='coerce')

fa = fa[fa['Flight_Month_dt'].between(window_start, window_end)]

miles_col  = next(c for c in ['Miles_Flown','Miles','miles','Distance','distance','Total_Miles'] if c in fa.columns)
spend_col  = next((c for c in ['Dollar_Proxy','Flight_Spend','Spend','spend','Dollars','Amount','Cost','Fare','Revenue'] if c in fa.columns), None)
flights_col = next((c for c in ['Flights','Flight_Count','flight_count','Total_Flights'] if c in fa.columns), None)

g = fa.groupby(['Loyalty_Number','Flight_Month_dt'])
cust_month = g.agg(
    Miles=(miles_col, 'sum'),
    Flights=(flights_col, 'sum') if flights_col else ('Loyalty_Number','size'),
    **({'Spend': (spend_col, 'sum')} if spend_col else {})
).reset_index()

panel = cust.assign(_k=1).merge(pd.DataFrame({'Flight_Month_dt': all_months, '_k': 1}), on='_k').drop(columns='_k')
panel = panel[panel['Flight_Month_dt'] >= panel['Enroll_Month_dt']].merge(cust_month, on=['Loyalty_Number','Flight_Month_dt'], how='left')

for c in ['Flights','Miles'] + (['Spend'] if spend_col else []):
    panel[c] = panel[c].fillna(0)

metrics = ['Flights','Miles'] + (['Spend'] if spend_col else [])

monthly_overall = (
    panel.groupby(['Flight_Month_dt','Promo_Flag'])
         .agg(Customers=('Loyalty_Number','nunique'), **{f'Avg_{m}': (m,'mean') for m in metrics})
         .reset_index()
)

monthly_by_band = (
    panel.groupby(['Flight_Month_dt','Promo_Flag','Salary_Band'])
         .agg(Customers=('Loyalty_Number','nunique'), **{f'Avg_{m}': (m,'mean') for m in metrics})
         .reset_index()
)


# In[39]:


print("monthly_by_band rows:", len(monthly_by_band))
print("monthly_by_band cols:", monthly_by_band.columns.tolist())
print(monthly_by_band[['Flight_Month_dt','Promo_Flag','Salary_Band']].drop_duplicates().head(10))

# ensure datetime
monthly_by_band['Flight_Month_dt'] = pd.to_datetime(monthly_by_band['Flight_Month_dt'], errors='coerce')

# force plot: Flights by Salary Band (Standard cohort)
fig, ax = plt.subplots(figsize=(12,6))

subset = monthly_by_band[monthly_by_band['Promo_Flag'] == False]

for band, df in subset.groupby('Salary_Band'):
    df = df.sort_values('Flight_Month_dt')
    ax.plot(df['Flight_Month_dt'], df['Avg_Flights'], marker='o', label=str(band))

ax.set_title("Avg Flights per Customer by Salary Band (Standard)")
ax.set_xlabel("Month")
ax.set_ylabel("Avg Flights")
ax.legend(title="Salary Band", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[40]:


print("monthly_by_band rows:", len(monthly_by_band))
print(monthly_by_band[['Flight_Month_dt','Promo_Flag','Salary_Band']].drop_duplicates().head(10))

monthly_by_band['Flight_Month_dt'] = pd.to_datetime(monthly_by_band['Flight_Month_dt'], errors='coerce')

fig, ax = plt.subplots(figsize=(12,6))

subset = monthly_by_band[monthly_by_band['Promo_Flag'] == False]

for band, df in subset.groupby('Salary_Band'):
    df = df.sort_values('Flight_Month_dt')
    ax.plot(df['Flight_Month_dt'], df['Avg_Miles'], marker='o', label=str(band))

ax.set_title("Avg Miles Flown per Customer by Salary Band (Standard)")
ax.set_xlabel("Month")
ax.set_ylabel("Avg Miles Flown")
ax.legend(title="Salary Band", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[41]:


monthly_overall['Flight_Month_dt'] = pd.to_datetime(monthly_overall['Flight_Month_dt'], errors='coerce')
monthly_by_band['Flight_Month_dt'] = pd.to_datetime(monthly_by_band['Flight_Month_dt'], errors='coerce')

# --- roll up to QUARTERLY ---
quarterly_overall = (
    monthly_overall
    .assign(Quarter=monthly_overall['Flight_Month_dt'].dt.to_period('Q').astype(str))
    .groupby(['Quarter', 'Promo_Flag'], as_index=False)
    .agg(
        Avg_Flights=('Avg_Flights', 'mean'),
        Avg_Miles=('Avg_Miles', 'mean'),
        **({'Avg_Spend': ('Avg_Spend', 'mean')} if 'Avg_Spend' in monthly_overall.columns else {})
    )
)

quarterly_by_band = (
    monthly_by_band
    .assign(Quarter=monthly_by_band['Flight_Month_dt'].dt.to_period('Q').astype(str))
    .groupby(['Quarter', 'Promo_Flag', 'Salary_Band'], as_index=False)
    .agg(
        Avg_Flights=('Avg_Flights', 'mean'),
        Avg_Miles=('Avg_Miles', 'mean'),
        **({'Avg_Spend': ('Avg_Spend', 'mean')} if 'Avg_Spend' in monthly_by_band.columns else {})
    )
)


# In[42]:


fig, ax = plt.subplots(figsize=(12, 6))
for flag, label in [(False, "Standard"), (True, "Promotion")]:
    tmp = quarterly_overall[quarterly_overall['Promo_Flag'] == flag].sort_values('Quarter')
    ax.plot(tmp['Quarter'], tmp['Avg_Flights'], marker='o', label=label)

ax.set_title("Avg Flights per Customer (Quarterly)")
ax.set_xlabel("Quarter")
ax.set_ylabel("Avg Flights")
ax.legend()
plt.tight_layout()
plt.show()


# In[43]:


fig, ax = plt.subplots(figsize=(12, 6))
for flag, label in [(False, "Standard"), (True, "Promotion")]:
    tmp = quarterly_overall[quarterly_overall['Promo_Flag'] == flag].sort_values('Quarter')
    ax.plot(tmp['Quarter'], tmp['Avg_Miles'], marker='o', label=label)

ax.set_title("Avg Miles Flown per Customer (Quarterly)")
ax.set_xlabel("Quarter")
ax.set_ylabel("Avg Miles")
ax.legend()
plt.tight_layout()
plt.show()


# In[44]:


fig, ax = plt.subplots(figsize=(12, 6))
tmp = quarterly_by_band[quarterly_by_band['Promo_Flag'] == False].copy()
for band, df in tmp.groupby('Salary_Band'):
    df = df.sort_values('Quarter')
    ax.plot(df['Quarter'], df['Avg_Flights'], marker='o', label=str(band))

ax.set_title("Avg Flights per Customer by Salary Band (Quarterly — Standard)")
ax.set_xlabel("Quarter")
ax.set_ylabel("Avg Flights")
ax.legend(title="Salary Band", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[45]:


fig, ax = plt.subplots(figsize=(12, 6))
tmp = quarterly_by_band[quarterly_by_band['Promo_Flag'] == True].copy()
for band, df in tmp.groupby('Salary_Band'):
    df = df.sort_values('Quarter')
    ax.plot(df['Quarter'], df['Avg_Flights'], marker='o', label=str(band))

ax.set_title("Avg Flights per Customer by Salary Band (Quarterly — Promotion)")
ax.set_xlabel("Quarter")
ax.set_ylabel("Avg Flights")
ax.legend(title="Salary Band", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[46]:


fig, ax = plt.subplots(figsize=(12, 6))
tmp = quarterly_by_band[quarterly_by_band['Promo_Flag'] == False].copy()
for band, df in tmp.groupby('Salary_Band'):
    df = df.sort_values('Quarter')
    ax.plot(df['Quarter'], df['Avg_Miles'], marker='o', label=str(band))

ax.set_title("Avg Miles Flown per Customer by Salary Band (Quarterly — Standard)")
ax.set_xlabel("Quarter")
ax.set_ylabel("Avg Miles")
ax.legend(title="Salary Band", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[47]:


fig, ax = plt.subplots(figsize=(12, 6))
tmp = quarterly_by_band[quarterly_by_band['Promo_Flag'] == True].copy()
for band, df in tmp.groupby('Salary_Band'):
    df = df.sort_values('Quarter')
    ax.plot(df['Quarter'], df['Avg_Miles'], marker='o', label=str(band))

ax.set_title("Avg Miles Flown per Customer by Salary Band (Quarterly — Promotion)")
ax.set_xlabel("Quarter")
ax.set_ylabel("Avg Miles")
ax.legend(title="Salary Band", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

