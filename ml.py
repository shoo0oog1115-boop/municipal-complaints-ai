import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

COLUMNS = ["complaint_text", "category"]
def clean_arabic_text(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"[^\u0621-\u064A\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def prepare(df):
    missing = set(COLUMNS) - set(df.columns)
    if missing: raise ValueError(f"الأعمدة المطلوبة غير موجودة: {sorted(missing)}")
    df = df[COLUMNS].dropna().copy()
    df[COLUMNS] = df[COLUMNS].astype(str)
    df['clean_text'] = df.complaint_text.map(clean_arabic_text)
    df = df[(df.clean_text != '') & (df.category.str.strip() != '')]
    df = df.drop_duplicates(subset=['clean_text', 'category']).reset_index(drop=True)
    if df.category.nunique() < 2: raise ValueError('يلزم وجود تصنيفين على الأقل')
    if df.category.value_counts().min() < 2: raise ValueError('يلزم مثالان على الأقل في كل تصنيف')
    return df

def make_model():
    return Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),max_features=50000,sublinear_tf=True)), ('classifier', LinearSVC(C=1.0,class_weight='balanced',random_state=42))])

def train_and_evaluate(df):
    df=prepare(df)
    if df.category.value_counts().min() < 5: raise ValueError('للتقسيم والتقييم يُنصح بخمسة أمثلة على الأقل لكل تصنيف')
    xtr, xte, ytr, yte = train_test_split(df.clean_text,df.category,test_size=.2,random_state=42,stratify=df.category)
    model=make_model();model.fit(xtr,ytr)
    pred=model.predict(xte)
    precision,recall,f1,_=precision_recall_fscore_support(yte,pred,average='macro',zero_division=0)
    metrics={'accuracy':accuracy_score(yte,pred),'precision_macro':precision,'recall_macro':recall,'f1_macro':f1,'training_rows':len(xtr),'test_rows':len(xte),'total_rows':len(df)}
    report=pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).T
    labels=sorted(df.category.unique());cm=pd.DataFrame(confusion_matrix(yte,pred,labels=labels),index=labels,columns=labels)
    final_model=make_model();final_model.fit(df.clean_text,df.category)
    return final_model,metrics,report,cm,df
