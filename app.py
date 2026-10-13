from pathlib import Path
import io
import pandas as pd
import streamlit as st
from ml import clean_arabic_text, train_and_evaluate

st.set_page_config(page_title="تصنيف البلاغات البلدية",page_icon="🏙️",layout="wide")
st.markdown("""<style>html,body,[class*=css]{direction:rtl;text-align:right} .block-container{padding-top:1.8rem} h1,h2,h3{color:#123C55}</style>""",unsafe_allow_html=True)
st.title("🏙️ نظام تصنيف البلاغات والشكاوى البلدية")
st.caption("نموذج تعلم آلي باللغة العربية | TF-IDF + LinearSVC | نموذج تجريبي تعليمي")
@st.cache_data
def load_default():
    return pd.read_csv(Path(__file__).parent/'municipal_complaints.csv')
@st.cache_resource(show_spinner="جارٍ تدريب النموذج وتحليل البيانات...")
def fit(_csv_bytes):
    df=pd.read_csv(io.BytesIO(_csv_bytes));return train_and_evaluate(df)
with st.sidebar:
    st.header("إعدادات البيانات")
    uploaded=st.file_uploader("تحميل CSV بديل",type=['csv'])
    st.info("الأعمدة المطلوبة: complaint_text و category")
    st.caption("البيانات الافتراضية اصطناعية لأغراض العرض والتجربة")
try:
    data=uploaded.getvalue() if uploaded else (Path(__file__).parent/'municipal_complaints.csv').read_bytes()
    model,metrics,report,cm,df=fit(data)
except Exception as exc:
    st.error(f"تعذر تجهيز البيانات أو النموذج: {exc}");st.stop()
page=st.tabs(["🔎 تصنيف شكوى","📊 لوحة المعلومات","🧪 تقييم النموذج","📁 البيانات والتصدير","ℹ️ عن المشروع"])
with page[0]:
    st.subheader("اكتب نص البلاغ")
    sample="الإنارة في الشارع الرئيسي معطلة منذ يومين ونرجو إصلاح الأعمدة"
    complaint=st.text_area("نص الشكوى",placeholder=sample,height=145)
    if st.button("تصنيف الشكوى",type="primary",use_container_width=True):
        cleaned=clean_arabic_text(complaint)
        if not cleaned:st.warning("يرجى إدخال نص عربي صالح قبل التصنيف")
        else:
            result=model.predict([cleaned])[0]
            st.success(f"التصنيف المتوقع: {result}")
            st.caption("هذا توقع آلي وليس قرارًا رسميًا. لا يعرض النموذج نسبة ثقة احتمالية معايرة.")
    st.divider();st.subheader("تصنيف عدة شكاوى")
    lines=st.text_area("ضع كل شكوى في سطر مستقل",height=120,key='bulk')
    if st.button("تصنيف القائمة"):
        rows=[x.strip() for x in lines.splitlines() if clean_arabic_text(x)]
        if not rows:st.warning("أدخل شكوى واحدة على الأقل")
        else:
            out=pd.DataFrame({'complaint_text':rows,'predicted_category':model.predict([clean_arabic_text(x) for x in rows])})
            st.dataframe(out,use_container_width=True)
            st.download_button("تنزيل النتائج CSV",out.to_csv(index=False).encode('utf-8-sig'),'predictions.csv','text/csv')
with page[1]:
    st.subheader("ملخص قاعدة البيانات")
    a,b,c=st.columns(3)
    a.metric("عدد السجلات بعد التنظيف",metrics['total_rows']);b.metric("عدد التصنيفات",df.category.nunique());c.metric("دقة الاختبار",f"{metrics['accuracy']:.1%}")
    st.bar_chart(df.category.value_counts());st.dataframe(df[['complaint_text','category']].head(15),use_container_width=True)
with page[2]:
    st.subheader("نتائج اختبار 20% من البيانات")
    a,b,c,d=st.columns(4)
    for col,k,label in zip([a,b,c,d],['accuracy','precision_macro','recall_macro','f1_macro'],['Accuracy','Precision macro','Recall macro','F1 macro']):col.metric(label,f"{metrics[k]:.1%}")
    st.caption(f"التدريب: {metrics['training_rows']} | الاختبار: {metrics['test_rows']} | تقسيم عشوائي ثابت مع المحافظة على نسب الفئات")
    st.write("تقرير التصنيف");st.dataframe(report,use_container_width=True)
    st.write("مصفوفة الالتباس — الصفوف فعلية والأعمدة متوقعة");st.dataframe(cm,use_container_width=True)
    st.warning("قد ترتفع النتائج على بيانات اصطناعية متشابهة؛ يجب الاختبار على شكاوى حقيقية مستقلة قبل أي استخدام عملي.")
with page[3]:
    st.subheader("استعراض البيانات")
    st.dataframe(df[['complaint_text','category']],use_container_width=True,height=420)
    st.download_button("تنزيل بيانات CSV",df[['complaint_text','category']].to_csv(index=False).encode('utf-8-sig'),'municipal_complaints_clean.csv','text/csv')
with page[4]:
    st.markdown("""**الفكرة:** توجيه الشكاوى البلدية إلى فئاتها تلقائيًا.  
**الطريقة:** تنظيف عربي ← TF-IDF للكلمات والثنائيات ← LinearSVC.  
**الحدود:** النموذج أولي، والفئات أربع، والبيانات التدريبية مصطنعة.  
**الخصوصية:** لا تدخل بيانات شخصية حقيقية في النسخة العامة.  
**التشغيل:** Streamlit Community Cloud باستخدام ملف `app.py` ومتطلبات `requirements.txt`.""")
