import os

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import warnings
from pathlib import Path
import re

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Lead Scoring Engine",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    try:
        model     = pickle.load(open('model.pkl',     'rb'))
        col_trans = pickle.load(open('col_trans.pkl', 'rb'))
        config    = json.load(open('threshold.json'))
        return model, col_trans, config, None
    except FileNotFoundError as e:
        return None, None, None, str(e)

model, col_trans, config, load_error = load_artefacts()

# ── Dropdown options ──────────────────────────────────────────────────────────
LEAD_SOURCES    = ['Google','Direct Traffic','Olark Chat','Organic Search',
                   'Reference','Welingak Website','Referral Sites','Facebook',
                   'bing','Click2call','Live Chat','Social Media',
                   'Pay per Click Ads','Press_Release','NC_EDM','blog']
LEAD_ORIGINS    = ['Landing Page Submission','Lead Add Form','Lead Import',
                   'Quick Add Form','API']
LAST_ACTIVITIES = ['Email Opened','SMS Sent','Olark Chat Conversation',
                   'Page Visited on Website','Converted to Lead',
                   'Email Bounced','Email Link Clicked',
                   'Form Submitted on Website','Had a Phone Conversation',
                   'View in Browser link Clicked','Visited Booth in Tradeshow',
                   'Approached upfront','Resubscribed to emails',
                   'Email Received','Modified']
LAST_NOTABLE    = ['Modified','Email Opened','Email Bounced','SMS Sent',
                   'Had a Phone Conversation','Olark Chat Conversation',
                   'Email Link Clicked','Page Visited on Website',
                   'View in Browser link Clicked','Resubscribed to emails']
SPECIALIZATIONS = ['Select','Finance Management','Human Resource Management',
                   'Marketing Management','Operations Management',
                   'Business Administration','IT Projects Management',
                   'Supply Chain Management','Banking, Investment And Insurance',
                   'Travel and Tourism','Media and Advertising',
                   'International Business','Healthcare Management',
                   'Hospitality Management','E-COMMERCE']
OCCUPATIONS     = ['Unemployed','Working Professional','Student',
                   'Businessman','Housewife','Other']
WHAT_MATTERS    = ['Better Career Prospects','Flexibility & Convenience',
                   'Other','Unknown']
CITIES          = ['Mumbai','Thane & Outskirts','Other Cities',
                   'Other Cities of Maharashtra','Other Metro Cities',
                   'Tier II Cities','outside India','Pune']
COUNTRIES       = ['India','Outside India','Unknown']

# ── Feature row + scoring ─────────────────────────────────────────────────────
def build_row(f):
    return pd.DataFrame([{
        'lead_origin'                                   : f['lead_origin'],
        'lead_source'                                   : f['lead_source'],
        'last_activity'                                 : f['last_activity'],
        'country'                                       : f['country'],
        'specialization'                                : f['specialization'],
        'what_is_your_current_occupation'               : f['occupation'],
        'what_matters_most_to_you_in_choosing_a_course' : f['what_matters'],
        'city'                                          : f['city'],
        'last_notable_activity'                         : f['last_notable'],
        'totalvisits'                                   : f['totalvisits'],
        'total_time_spent_on_website'                   : f['total_time'],
        'page_views_per_visit'                          : f['page_views'],
        'do_not_email'                                  : f['do_not_email'],
        'do_not_call'                                   : f['do_not_call'],
        'a_free_copy_of_mastering_the_interview'        : f['free_copy'],
    }])

def score_lead(f):
    prob = model.predict_proba(col_trans.transform(build_row(f)))[0][1]
    s    = round(prob * 100, 1)
    tier = "HOT" if s >= 80 else ("WARM" if s >= 50 else "COLD")
    return s, prob, tier

def get_shap(f):
    import shap as _shap
    row_t = col_trans.transform(build_row(f))
    names = [
        n.replace('category__','').replace('numerical_skew__','')
         .replace('bin_col__','').replace('_',' ').title()
        for n in col_trans.get_feature_names_out()
    ]
    base = model.estimator if hasattr(model, 'estimator') else model
    sv   = _shap.TreeExplainer(base)(pd.DataFrame(row_t, columns=names))
    return sv.values[0], names

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("Lead Scoring Engine")
st.caption("SHAP explanability  -  XGBoost - CRM Lead Scoring - https://github.com/shikharuniyal")
st.divider()

# ── Error guard ───────────────────────────────────────────────────────────────
if load_error:
    st.error(
        "Model loading issue"
    )
    st.stop()

threshold = config.get('optimal_threshold', 0.65)

# ══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT — FORM
# ─────────────────────────────────────────────────────────────────────────────
with left:

    with st.container(border=True):
        st.caption("ACQUISITION of the Lead")
        c1, c2 = st.columns(2)
        with c1:
            lead_source = st.selectbox("Lead Source", LEAD_SOURCES)
        with c2:
            lead_origin = st.selectbox("Lead Origin", LEAD_ORIGINS)

    with st.container(border=True):
        st.caption("BEHAVIOUR PATTERN")
        n1, n2 = st.columns(2)
        with n1:
            total_time  = st.number_input(
                "Time on Website-sec", min_value=0,
                max_value=10000, value=800, step=50
            )
        with n2:
            totalvisits = st.number_input(
                "Total Visits", min_value=0,
                max_value=200, value=5, step=1
            )
        page_views    = st.slider(
            "Page Views Per Visit", min_value=0.0,
            max_value=20.0, value=3.0, step=0.5
        )
        last_activity = st.selectbox("Last Activity", LAST_ACTIVITIES)
        last_notable  = st.selectbox("Last Notable Activity", LAST_NOTABLE)

    with st.container(border=True):
        st.caption("USER PROFILE")
        p1, p2 = st.columns(2)
        with p1:
            occupation     = st.selectbox("Occupation",     OCCUPATIONS)
            specialization = st.selectbox("Specialization", SPECIALIZATIONS)
        with p2:
            city    = st.selectbox("City",    CITIES)
            country = st.selectbox("Country", COUNTRIES)
        what_matters = st.selectbox("What Matters Most To Them?", WHAT_MATTERS)

    with st.container(border=True):
        st.caption("SELECTED CONTACT PREFERENCES")
        x1, x2, x3 = st.columns(3)
        with x1:
            do_not_email = st.selectbox("Do Not Email",        ["No", "Yes"])
        with x2:
            do_not_call  = st.selectbox("Do Not Call",         ["No", "Yes"])
        with x3:
            free_copy    = st.selectbox("Requested Free Copy", ["No", "Yes"])

    score_btn = st.button("Score The Lead", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with right:

    if not score_btn:
        st.info(
            "Fill in the lead details "
            "To **Score The Lead**."
        )

    else:
        inputs = dict(
            lead_source=lead_source, lead_origin=lead_origin,
            last_activity=last_activity, country=country,
            specialization=specialization, occupation=occupation,
            what_matters=what_matters, city=city,
            last_notable=last_notable, totalvisits=totalvisits,
            total_time=total_time, page_views=page_views,
            do_not_email=do_not_email, do_not_call=do_not_call,
            free_copy=free_copy,
        )

        try:
            with st.spinner("Scoring lead..."):
                s, prob, tier = score_lead(inputs)
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            st.stop()

        tier_action = {
            "HOT" : "High Priority",
            "WARM": "Potentional Lead",
            "COLD": "Low Priority"
        }[tier]

        

        # ── Gauge ─────────────────────────────────────────────────────────────
        fig_g = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = s,
            number= {
                'suffix'     : ' / 100',
                'valueformat': '.1f',
                'font'       : {'size': 46, 'color': "#000000"}
            },
            gauge = {
                'axis'       : {'range': [0, 100]},
                'bar'        : {'color': "#46cf9d", 'thickness': 0.22},
                'bgcolor'    : "#004e39",
                'bordercolor': "#63936a",
                'borderwidth': 1,
            }
        ))
        fig_g.update_layout(
            height=210,
            margin=dict(t=20, b=0, l=30, r=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor ='rgba(0,0,0,0)',
        )
        st.plotly_chart(
            fig_g,
            use_container_width=True,
            config={'displayModeBar': False}
        )

        # ── Tier result ───────────────────────────────────────────────────────
        with st.container(border=True):
            st.subheader(f"{tier}  —  {tier_action}")
            st.write(f"**P(convert) = {prob:.4f}**  ·  Threshold = {threshold:.2f}")

        st.divider()

        # ── Model stats ───────────────────────────────────────────────────────
        auc   = config.get('auc_roc_calibrated', config.get('auc_roc_raw', 0))
        brier = config.get('brier_score', 0)
        f1    = config.get('f1_at_optimal', 0)

        m1, m2, m3 = st.columns(3)
        m1.metric(label="AUC-ROC",     value=f"{auc:.3f}",   border=True)
        m2.metric(label="Brier Score", value=f"{brier:.3f}", border=True)
        m3.metric(label="F1 Score",    value=f"{f1:.3f}",    border=True)

        st.divider()

        # ── Feature contributions ─────────────────────────────────────────────
        st.caption("FEATURE CONTRIBUTIONS")

        try:
            with st.spinner("Computing feature contributions..."):
                shap_vals, feat_names = get_shap(inputs)

            top_idx  = np.argsort(np.abs(shap_vals))[-10:][::-1]
            sv_top   = shap_vals[top_idx]
            fn_top   = [feat_names[i][:40] for i in top_idx]

            chart_df = pd.DataFrame(
                {'Contribution': sv_top},
                index=fn_top
            )

            st.bar_chart(
                chart_df,
                horizontal=True,
                use_container_width=True,
                height=320,
                color="#2d6a4f"
            )

            st.caption(
                "Positive values increase the score  ·  "
                "Negative values decrease the score"
            )

        except Exception as e:
            st.warning(f"Feature contributions unavailable: {str(e)[:100]}")


st.divider()


#result explanation with image

ASSETS_DIR = Path(__file__).parent / "assets"

def plot_number(path):
    match = re.search(r"plot(\d+)", path.stem.lower())
    return int(match.group(1)) if match else 9999

image_files = sorted(
    [
        f for f in ASSETS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    ],
    key=plot_number
)

DESCRIPTIONS = [

    #plot1
    {
        "title": "Numerical Feature Correlations",
        "description": (
            "Total Visits and Page Views Per Visit are nearly uncorrelated with each other "
            "and with Total Time on Website. This means all three numerical features are "
            "measuring different things — none are redundant. Total Time on Website is "
            "independently the strongest signal the model has."
        )
    },

    #plot2
    {
        "title": "Class Imbalance",
        "description": (
            "5,672 leads did not convert versus 3,532 that did gives a 61.6 / 38.4 split. "
            "The dataset is moderately imbalanced at a 1.61:1 ratio. This is why the model "
            "uses scale_pos_weight = 1.61: without it, XGBoost would learn to predict "
            "not converted by default and still appear 62% accurate."
        )
    },

    #plot3
    {
        "title": "Numerical Distributions by Conversion Outcome",
        "description": (
            "Total Time on Website shows the clearest separation converted leads "
            "cluster at higher time values while non-converters drop off early. "
            "Total Visits and Page Views Per Visit have heavily right-skewed distributions "
            "with extreme outliers (skewness of 20 and 2.9 respectively), confirming why "
            "PowerTransformer was applied before training."
        )
    },

    #plot4
    {
        "title": "Conversion Rate by Category",
        "description": (
            "Leads from Welingak Website and Reference sources convert at well above "
            "the dataset average, while Facebook and bing traffic converts poorly. "
            "Working Professionals convert at the highest rate among occupations. "
            "Leads whose last activity was a Phone Conversation convert at near double "
            "the rate of those whose last activity was Email Bounced."
        )
    },
    # plot5

    {
        "title": "Mutual Information vs Target",
        "description": (
            "Total Time on Website (MI = 0.121) is the single most informative feature "
            "by a clear margin ,it alone explains more variance in conversion outcome "
            "than any other signal. Last Activity (0.082) and Last Notable Activity (0.076) "
            "follow closely. Page Views Per Visit (below 0.01) and Free Copy Requested "
            "(0.007) are near-noise and contribute almost nothing to predicting conversion."
        )
    },
    # plot6
    {
        "title": "XGBoost Feature Importance (top 10 features only)",
        "description": (
            "The model leans most heavily on Total Time on Website when making splits "
            "across all 100 trees that are consistent with its top MI score. Last Activity "
            "and Lead Source follow. The three binary contact preference features "
            "(Do Not Email, Do Not Call, Free Copy) rank low, confirming they carry "
            "limited predictive signal despite being useful as filters."
        )
    },

    #plot7
    {
        "title": "Confusion Matrix",
        "description": (
            "On 1,841 test leads: 184 were wrongly flagged as likely converters (False "
            "Positives) — wasted sales calls. 123 real converters were missed (False "
            "Negatives) — lost revenue opportunities. Overall accuracy is 83%, but the "
            "model is slightly better at identifying non-converters (84% recall) than "
            "converters (83% recall)."
        )
    },

    #plot8
    {
        "title": "ROC and Precision-Recall Curves",
        "description": (
            "AUC-ROC of 0.905 — the model correctly ranks a random converter above a "
            "random non-converter 90.5% of the time, versus 50% for a random baseline. "
            "The Precision-Recall curve stays well above the 0.38 random baseline "
            "across most recall values, meaning the model can find the majority of "
            "real converters while keeping false alarms low."
        )
    },

    #plot9
    {
        "title": "Calibration Curve",
        "description": (
            "Raw XGBoost is overconfident — it pushes predictions toward 0 and 1 more "
            "than the data justifies. After Platt scaling, the curve moves closer to the "
            "perfect diagonal. The Brier Score improves from 0.1226 to 0.1209. The gain "
            "is modest but meaningful: the calibrated model's P(convert) = 0.72 now "
            "genuinely reflects a ~72% historical conversion rate for that lead profile."
        )
    },

    #plot10
    {
        "title": "Threshold Optimisation",
        "description": (
            "The default threshold of 0.50 gives F1 = 0.7916. The optimal threshold "
            "of 0.49 gives F1 = 0.7930  a marginal improvement, meaning the dataset "
            "is fairly balanced around that decision boundary. The F1 curve is flat near "
            "the peak, so the model is not sensitive to small threshold changes between "
            "0.45 and 0.55."
        )
    },

    #plot11
    {
        "title": "Lead Score Distribution",
        "description": (
            "The score distribution is bimodal — a large cluster of leads scoring below "
            "30 (clear non-converters) and a second cluster scoring above 70 (high-intent "
            "leads). Relatively few leads sit in the 40–60 ambiguous zone, which means "
            "the model is making confident decisions on most leads rather than sitting "
            "on the fence."
        )
    },

    #plot12
    {
        "title": "SHAP Global Feature Importance",
        "description": (
            "Total Time on Website has the largest average impact on predictions across "
            "all test leads — it moves scores more than any other single feature. "
            "Last Activity and Last Notable Activity together account for the next largest "
            "share of the model's output. The bottom features (binary flags like "
            "Do Not Call) barely shift predictions at all for most leads."
        )
    },

    #plot13
    {
        "title": "SHAP Beeswarm (Feature Direction)",
        "description": (
            "High time on website (red dots) consistently pushes scores to the right — "
            "increasing conversion probability. Low time (blue dots) pulls scores down. "
            "The effect is monotonic and strong. Last Activity shows a wide spread "
            "of both red and blue dots on both sides of zero, meaning its direction "
            "of impact depends entirely on which activity it was — phone call pushes "
            "up, email bounce pushes down."
        )
    },

    #plot14
    {
        "title": "SHAP Dependence (top 3 features)",
        "description": (
            "Total Time on Website shows a clear non-linear step: below roughly 400 "
            "seconds the SHAP value is negative (hurts conversion), above it the value "
            "rises steeply, then flattens after approx. 1,500 seconds i.e. time beyond that point "
            "adds diminishing signal. This threshold effect is why a tree-based model "
            "outperforms logistic regression on this dataset."
        )
    },

    #plot15
    {
        "title": "Post-Encoding Correlation — Top 15 Features",
        "description": (
            "After one-hot encoding and Yeo-Johnson transformation, the top 15 features "
            "are largely uncorrelated with each other. No pair exceeds 0.4 correlation, "
            "which means the encoding process did not introduce multicollinearity. "
            "The model is working with 15 genuinely independent signals, not redundant "
            "copies of the same information in different column names."
        )
    },
    

    #plot16
    {
        "title": "XGBoost Decision Tree ",
        "description": (
            "The first split in Tree 0 is on Total Time on Website, confirming it as "
            "the model's primary discriminator. Leads spending above the split threshold "
            "immediately route toward higher leaf scores. Subsequent splits on Last "
            "Activity and Lead Source refine the prediction. This single tree is a weak "
            "learner — the final score comes from summing 100 such trees, each one "
            "correcting the residual errors left by the previous."
        )
    },

    #plot17
   
     
    {
        "title": "Cold Lead (Lowest Score)",
        "description": (
            "The prediction is strongly negative, driven by email opt-out, low time on site, student status, and lack of specialization. Nearly all major features reduce the conversion score, indicating very low engagement and conversion likelihood. "
        )
    },
    {
        "title": "Top Lead (Highest Score)",
        "description": (
            "The prediction is strongly positive, driven by Lead Add Form origin, Working Professional status, and repeated SMS engagement signals. Despite slightly lower time on site, the combined positive signals indicate a highly engaged, sales-ready lead."
        )
    },
    {
        "title": "SHAP Waterfall — Individual Lead Breakdown",
        "description": (
            "For the top-scoring lead, high time on site and a phone conversation as "
            "last activity together account for most of the score lift above baseline. "
            "For the lowest-scoring lead, near-zero time on site combined with an "
            "email bounce drives the prediction well below baseline. The borderline lead "
            "near score 50 has competing signals roughly cancelling each other out — "
            "moderate time on site pushing up, a weak lead source pulling down."
        )
    },
    
]

st.divider()
st.header("Model Performance Analysis")

for i, image_file in enumerate(image_files):
    st.divider()

    info = DESCRIPTIONS[i] if i < len(DESCRIPTIONS) else {
        "title": image_file.stem.replace("_", " ").title(),
        "description": "Add description here."
    }

    col_img, col_text = st.columns([1.4, 1], gap="large")

    with col_img:
        st.image(image_file, use_container_width=True)

    with col_text:
        st.subheader(info["title"])
        st.write(info["description"])

        
st.divider()
st.caption(
    f"Made By Shikhar Uniyal  ·  "
    f"{config.get('training_rows', '—')} leads  ·  "
    f"{config.get('n_features', '—')} features  ·  "
    f"XGBoost  ·  "
    f"CRM Lead analysis Demonstraiton"
    f"dataset Lead X Education"
    f"\nGithub repo: https://github.com/shikharuniyal/Lead-Navigator"
)