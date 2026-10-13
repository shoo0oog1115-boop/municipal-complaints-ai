from pathlib import Path
import io
import base64
import html
import pandas as pd
import streamlit as st
from ml import clean_arabic_text, train_and_evaluate

st.set_page_config(page_title="منصة المدينة الذكية | البلاغات البلدية", page_icon="🕌", layout="wide", initial_sidebar_state="collapsed")
css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# الصورة موجودة مسبقاً في جذر المستودع بالاسم التالي.
image_path = Path(__file__).parent / "IMG_7568.jpg"
if image_path.is_file():
    image_uri = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_css = f'linear-gradient(110deg, rgba(5,19,20,.72), rgba(6,29,27,.78)), url("{image_uri}")'
else:
    image_css = 'linear-gradient(125deg, #061b1b, #17483b)'

st.markdown(f"""
<style>
/* خلفية المدينة المنورة بكامل الصفحة */
.stApp {{
    background-image: {image_css};
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    color: #f8f3e7;
}}
html, body, [data-testid="stAppViewContainer"] {{ direction: rtl; }}
[data-testid="stAppViewContainer"] .main .block-container {{
    max-width: 1080px;
    padding: 1.5rem 2rem 3rem;
    margin-top: 1.2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(215,183,113,.42);
    border-radius: 24px;
    background: rgba(5,25,24,.78);
    box-shadow: 0 18px 55px rgba(0,0,0,.38);
    backdrop-filter: blur(10px);
}}
/* العناوين والنصوص */
h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {{
    color: #f9f5eb;
    text-align: right;
}}
h1 {{ font-size: clamp(1.65rem, 3vw, 2.5rem) !important; line-height: 1.5 !important; }}
h2, h3 {{ color: #e6c780 !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: #dfd4b8 !important; }}
/* الشريط الجانبي */
section[data-testid="stSidebar"] {{
    background: #092723;
    border-left: 1px solid #9c8150;
}}
section[data-testid="stSidebar"] * {{ color: #f5edda; }}
/* التبويبات */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(0,0,0,.23);
    border: 1px solid rgba(212,176,106,.3);
    border-radius: 14px;
    padding: 8px;
    flex-wrap: wrap;
}}
.stTabs [data-baseweb="tab"] {{
    color: #eee4ce;
    background: rgba(255,255,255,.06);
    border-radius: 10px;
    padding: 8px 13px;
}}
.stTabs [aria-selected="true"] {{
    color: #092d26 !important;
    background: #e3c37c !important;
    font-weight: 700;
}}
/* أزرار خضراء وذهبية */
.stButton button, .stDownloadButton button {{
    color: #fff8e8 !important;
    background: linear-gradient(110deg, #11614d, #0b453b) !important;
    border: 1px solid #dfbd72 !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
    min-height: 44px;
    box-shadow: 0 3px 10px rgba(0,0,0,.2);
}}
.stButton button:hover, .stDownloadButton button:hover {{
    color: #102b25 !important;
    background: #e7c77f !important;
}}
/* مدخلات بيضاء لسهولة القراءة */
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {{
    background: rgba(255,252,245,.96) !important;
    color: #17352e !important;
    border: 1px solid #c9ad76 !important;
    border-radius: 11px !important;
    text-align: right;
    direction: rtl;
}}
[data-testid="stTextArea"] textarea::placeholder {{ color: #60736c !important; }}
[data-testid="stMetric"] {{
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(225,190,115,.45);
    border-radius: 13px;
    padding: 12px;
}}
[data-testid="stMetric"] * {{ color: #f9efd9 !important; }}
[data-testid="stAlert"] {{ border-radius: 12px; }}
hr {{ border-color: rgba(227,195,124,.35) !important; }}
/* الجداول تبقى بخلفية مقروءة */
[data-testid="stDataFrame"] {{ border: 1px solid rgba(225,190,115,.45); border-radius: 10px; }}
@media (max-width: 720px) {{
    [data-testid="stAppViewContainer"] .main .block-container {{
        padding: 1rem .8rem 2rem;
        margin-top: .5rem;
    }}
}}
</style>
""", unsafe_allow_html=True)

st.title("🕌 منصة المدينة الذكية لتصنيف البلاغات والشكاوى البلدية")
st.caption("تصنيف البلاغات العربية باستخدام تعلم الآلة | TF-IDF + LinearSVC | نموذج تجريبي تعليمي")
st.markdown("---")

@st.cache_data
def load_default():
    return pd.read_csv(Path(__file__).parent / 'municipal_complaints.csv')

@st.cache_resource(show_spinner="جارٍ تدريب النموذج وتحليل البيانات...")
def fit(_csv_bytes):
    df = pd.read_csv(io.BytesIO(_csv_bytes))
    return train_and_evaluate(df)

with st.sidebar:
    st.header("⚙️ إعدادات البيانات")
    uploaded = st.file_uploader("تحميل CSV بديل", type=['csv'])
    st.info("الأعمدة المطلوبة: complaint_text و category")
    st.caption("البيانات الافتراضية اصطناعية لأغراض العرض والتجربة")

try:
    data = uploaded.getvalue() if uploaded else (Path(__file__).parent / 'municipal_complaints.csv').read_bytes()
    model, metrics, report, cm, df = fit(data)
except Exception as exc:
    st.error(f"تعذر تجهيز البيانات أو النموذج: {exc}")
    st.stop()

pages = st.tabs(["🔎 تصنيف شكوى", "📊 لوحة المعلومات", "🧪 تقييم النموذج", "📁 البيانات والتصدير", "ℹ️ عن المشروع"])

with pages[0]:
    st.subheader("✍️ اكتب نص البلاغ")
    sample = "الإنارة في الشارع الرئيسي معطلة منذ يومين ونرجو إصلاح الأعمدة"
    complaint = st.text_area("نص الشكوى", placeholder=sample, height=140)
    if st.button("🔎 تصنيف الشكوى", type="primary", use_container_width=True):
        cleaned = clean_arabic_text(complaint)
        if not cleaned:
            st.warning("يرجى إدخال نص عربي صالح قبل التصنيف")
        else:
            result = model.predict([cleaned])[0]
            st.success(f"التصنيف المتوقع: {result}")
            st.caption("هذا توقع آلي وليس قرارًا رسميًا. لا يعرض النموذج نسبة ثقة احتمالية معايرة.")
    st.divider()
    st.subheader("📋 تصنيف عدة شكاوى")
    lines = st.text_area("ضع كل شكوى في سطر مستقل", height=120, key="bulk")
    if st.button("تصنيف القائمة"):
        rows = [x.strip() for x in lines.splitlines() if clean_arabic_text(x)]
        if not rows:
            st.warning("أدخل شكوى واحدة على الأقل")
        else:
            out = pd.DataFrame({
                'complaint_text': rows,
                'predicted_category': model.predict([clean_arabic_text(x) for x in rows])
            })
            st.dataframe(out, use_container_width=True)
            st.download_button("⬇️ تنزيل النتائج CSV", out.to_csv(index=False).encode('utf-8-sig'), 'predictions.csv', 'text/csv')

with pages[1]:
    st.subheader("📊 ملخص قاعدة البيانات")
    a, b, c = st.columns(3)
    a.metric("عدد السجلات بعد التنظيف", metrics['total_rows'])
    b.metric("عدد التصنيفات", df.category.nunique())
    c.metric("دقة الاختبار", f"{metrics['accuracy']:.1%}")
    st.bar_chart(df.category.value_counts())
    st.dataframe(df[['complaint_text', 'category']].head(15), use_container_width=True)

with pages[2]:
    st.subheader("🧪 نتائج اختبار 20% من البيانات")
    a, b, c, d = st.columns(4)
    for col, key, label in zip(
        [a, b, c, d],
        ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        ['Accuracy', 'Precision macro', 'Recall macro', 'F1 macro']
    ):
        col.metric(label, f"{metrics[key]:.1%}")
    st.caption(f"التدريب: {metrics['training_rows']} | الاختبار: {metrics['test_rows']} | تقسيم عشوائي ثابت مع المحافظة على نسب الفئات")
    st.write("تقرير التصنيف")
    st.dataframe(report, use_container_width=True)
    st.write("مصفوفة الالتباس — الصفوف فعلية والأعمدة متوقعة")
    st.dataframe(cm, use_container_width=True)
    st.warning("قد ترتفع النتائج على بيانات اصطناعية متشابهة؛ يجب الاختبار على شكاوى حقيقية مستقلة قبل أي استخدام عملي.")

with pages[3]:
    st.subheader("📁 استعراض البيانات")
    st.dataframe(df[['complaint_text', 'category']], use_container_width=True, height=420)
    st.download_button(
        "⬇️ تنزيل بيانات CSV",
        df[['complaint_text', 'category']].to_csv(index=False).encode('utf-8-sig'),
        'municipal_complaints_clean.csv', 'text/csv'
    )

with pages[4]:
    st.markdown("""**الفكرة:** توجيه الشكاوى البلدية إلى فئاتها تلقائيًا.  
**الطريقة:** تنظيف عربي ← TF-IDF للكلمات والثنائيات ← LinearSVC.  
**الحدود:** النموذج أولي، والفئات أربع، والبيانات التدريبية مصطنعة.  
**الخصوصية:** لا تدخل بيانات شخصية حقيقية في النسخة العامة.  
**التشغيل:** Streamlit Community Cloud باستخدام ملف `app.py` ومتطلبات `requirements.txt`.""")
