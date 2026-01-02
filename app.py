import streamlit as st
import openai
import os
import tempfile
import pandas as pd
import plotly.express as px
from datetime import datetime
import io

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Dashboard تحليل المكالمات",
    page_icon="📊",
    layout="wide"
)

# --- التعامل مع المفاتيح السرية (Secrets) ---
# يحاول البرنامج قراءة المفتاح من إعدادات السيرفر، إذا لم يجده يطلبه من المستخدم
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    # الشريط الجانبي في حالة عدم وجود مفتاح مخزن
    with st.sidebar:
        api_key = st.text_input("أدخل OpenAI API Key", type="password")
        if not api_key:
            st.warning("⚠️ أدخل المفتاح ليعمل البرنامج")

# --- التنسيق الجمالي ---
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- الدوال المساعدة ---
def transcribe_audio(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        return client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text"
        )

def analyze_call_data(client, text):
    """تحليل لاستخراج بيانات مهيكلة للرسوم البيانية"""
    prompt = f"""
    حلل نص المكالمة التالي واستخرج البيانات بصيغة JSON فقط:
    1. "score": رقم من 1 إلى 10 لتقييم الموظف.
    2. "sentiment": (Positive, Negative, Neutral).
    3. "topic": موضوع المكالمة في كلمة أو كلمتين (مثلاً: فاتورة، عطل فني، شكوى).
    4. "summary": ملخص عربي في سطر واحد.
    
    النص: "{text}"
    Output Format: {{"score": 8, "sentiment": "Positive", "topic": "Billing", "summary": "..."}}
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": "You are a data extractor. Output JSON only."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    import json
    return json.loads(response.choices[0].message.content)

def detailed_feedback(client, text):
    prompt = f"""
    قم بتقديم نقد بناء للموظف بناء على المكالمة:
    - نقاط القوة.
    - نقاط تحتاج لتحسين.
    - هل التزم بآداب الحديث؟
    النص: {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- التطبيق الرئيسي ---
st.title("📊 نظام ذكاء الأعمال لمراكز الاتصال (AI Call Analysis)")

if not api_key:
    st.info("👈 يرجى إدخال مفتاح API في القائمة الجانبية أو إعداده في Secrets.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

uploaded_files = st.file_uploader("ارفع ملفات المكالمات (MP3, WAV, M4A)", 
                                  type=['mp3', 'wav', 'm4a'], accept_multiple_files=True)

if uploaded_files:
    if st.button(f"🚀 بدء تحليل {len(uploaded_files)} مكالمات"):
        
        results = []
        progress_bar = st.progress(0)
        st.subheader("📝 تفاصيل المكالمات")
        
        for idx, file in enumerate(uploaded_files):
            try:
                with st.spinner(f"جاري تحليل: {file.name}..."):
                    # حفظ مؤقت
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    # العمليات
                    transcript = transcribe_audio(client, tmp_path)
                    data_points = analyze_call_data(client, transcript)
                    feedback = detailed_feedback(client, transcript)
                    
                    # تجميع البيانات
                    call_record = {
                        "اسم الملف": file.name,
                        "التقييم (10)": data_points.get('score', 0),
                        "المشاعر": data_points.get('sentiment', 'Neutral'),
                        "الموضوع": data_points.get('topic', 'General'),
                        "الملخص": data_points.get('summary', ''),
                        "النص الكامل": transcript,
                        "التقرير التفصيلي": feedback
                    }
                    results.append(call_record)
                    
                    # عرض سريع
                    with st.expander(f"📞 {file.name} - {data_points.get('score')}/10"):
                        st.write(f"**الموضوع:** {data_points.get('topic')}")
                        st.info(feedback)
                    
                    os.remove(tmp_path)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
            except Exception as e:
                st.error(f"حدث خطأ في الملف {file.name}: {e}")

        # --- Dashboard ---
        if results:
            st.markdown("---")
            st.header("📈 لوحة التحليلات المجمعة")
            df = pd.DataFrame(results)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("متوسط الأداء", f"{df['التقييم (10)'].mean():.1f}/10")
            col2.metric("عدد المكالمات", len(df))
            col3.metric("أقل تقييم", df["التقييم (10)"].min())

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("المشاعر")
                fig = px.pie(df, names='المشاعر', color='المشاعر', 
                             color_discrete_map={'Positive':'#4CAF50', 'Negative':'#EF5350', 'Neutral':'#FFC107'})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("المواضيع")
                fig = px.bar(df, x='الموضوع', y='التقييم (10)')
                st.plotly_chart(fig, use_container_width=True)

            # التصدير
            def to_excel(df):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                return output.getvalue()
                
            st.download_button("📥 تحميل التقرير (Excel)", data=to_excel(df), 
                               file_name='Report.xlsx', 
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')