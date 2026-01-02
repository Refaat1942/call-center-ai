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
    page_title="Lotus Calls Quality",
    page_icon="🎧",
    layout="wide"
)

# --- النصوص ثنائية اللغة ---
translations = {
    'en': {
        'app_title': 'Lotus Calls Quality Analysis',
        'sidebar_title': 'Settings & Info',
        'lang_select': 'Language / اللغة',
        'api_key_label': 'Enter OpenAI API Key',
        'api_key_warning': '⚠️ Enter API Key to run the app',
        'api_key_info': '👈 Please enter API Key in the sidebar or set it in Secrets.',
        'file_uploader_label': 'Upload Call Files (MP3, WAV, M4A)',
        'start_analysis_btn': '🚀 Start Analysis of {len} calls',
        'call_details_header': '📝 Call Details',
        'analyzing_spinner': 'Analyzing: {file_name}...',
        'call_expander': '📞 {file_name} - {score}/10',
        'topic_label': '**Topic:**',
        'dashboard_header': '📈 Consolidated Analysis Dashboard',
        'metric_avg_score': 'Average Score',
        'metric_call_count': 'Total Calls',
        'metric_min_score': 'Lowest Score',
        'chart_sentiment': 'Sentiment Distribution',
        'chart_topics': 'Call Topics & Scores',
        'download_btn': '📥 Download Report (Excel)',
        'error_msg': 'Error in file {file_name}: {error}',
    },
    'ar': {
        'app_title': 'نظام تحليل جودة مكالمات لوتس',
        'sidebar_title': 'الإعدادات والمعلومات',
        'lang_select': 'اللغة / Language',
        'api_key_label': 'أدخل OpenAI API Key',
        'api_key_warning': '⚠️ أدخل المفتاح ليعمل البرنامج',
        'api_key_info': '👈 يرجى إدخال مفتاح API في القائمة الجانبية أو إعداده في Secrets.',
        'file_uploader_label': 'ارفع ملفات المكالمات (MP3, WAV, M4A)',
        'start_analysis_btn': '🚀 بدء تحليل {len} مكالمات',
        'call_details_header': '📝 تفاصيل المكالمات',
        'analyzing_spinner': 'جاري تحليل: {file_name}...',
        'call_expander': '📞 {file_name} - {score}/10',
        'topic_label': '**الموضوع:**',
        'dashboard_header': '📈 لوحة التحليلات المجمعة',
        'metric_avg_score': 'متوسط الأداء',
        'metric_call_count': 'عدد المكالمات',
        'metric_min_score': 'أقل تقييم',
        'chart_sentiment': 'توزيع المشاعر',
        'chart_topics': 'مواضيع المكالمات والتقييم',
        'download_btn': '📥 تحميل التقرير (Excel)',
        'error_msg': 'حدث خطأ في الملف {file_name}: {error}',
    }
}

# --- الشريط الجانبي وتحديد اللغة ---
with st.sidebar:
    # عرض اللوجو
    st.image("image_5.png", use_container_width=True) #
    
    st.title(translations['ar']['sidebar_title'])
    
    # اختيار اللغة
    lang = st.selectbox(translations['ar']['lang_select'], ('Arabic', 'English'))
    lang_code = 'ar' if lang == 'Arabic' else 'en'
    t = translations[lang_code] # تحديد مجموعة النصوص

    # التعامل مع مفتاح API
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input(t['api_key_label'], type="password")
        if not api_key:
            st.warning(t['api_key_warning'])

# --- التنسيق الجمالي ---
st.markdown(f"""
<style>
    .stApp {{background-color: #f4f7f6;}}
    h1 {{ color: #004d40; }}
    .metric-container {{
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

# --- الدوال المساعدة ---
def transcribe_audio(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        return client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text"
        )

def analyze_call_data(client, text):
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
st.title(t['app_title']) # استخدام العنوان حسب اللغة

if not api_key:
    st.info(t['api_key_info'])
    st.stop()

client = openai.OpenAI(api_key=api_key)

uploaded_files = st.file_uploader(t['file_uploader_label'], 
                                  type=['mp3', 'wav', 'm4a'], accept_multiple_files=True)

if uploaded_files:
    if st.button(t['start_analysis_btn'].format(len=len(uploaded_files))):
        
        results = []
        progress_bar = st.progress(0)
        st.subheader(t['call_details_header'])
        
        for idx, file in enumerate(uploaded_files):
            try:
                with st.spinner(t['analyzing_spinner'].format(file_name=file.name)):
                    # حفظ مؤقت
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    # العمليات
                    transcript = transcribe_audio(client, tmp_path)
                    data_points = analyze_call_data(client, transcript)
                    feedback = detailed_feedback(client, transcript)
                    
                    score = data_points.get('score', 0)
                    # تجميع البيانات
                    call_record = {
                        "اسم الملف" if lang_code == 'ar' else "File Name": file.name,
                        "التقييم (10)" if lang_code == 'ar' else "Score (10)": score,
                        "المشاعر" if lang_code == 'ar' else "Sentiment": data_points.get('sentiment', 'Neutral'),
                        "الموضوع" if lang_code == 'ar' else "Topic": data_points.get('topic', 'General'),
                        "الملخص" if lang_code == 'ar' else "Summary": data_points.get('summary', ''),
                        "النص الكامل" if lang_code == 'ar' else "Full Transcript": transcript,
                        "التقرير التفصيلي" if lang_code == 'ar' else "Detailed Report": feedback
                    }
                    results.append(call_record)
                    
                    # عرض سريع
                    with st.expander(t['call_expander'].format(file_name=file.name, score=score)):
                        st.write(f"{t['topic_label']} {data_points.get('topic')}")
                        st.info(feedback)
                    
                    os.remove(tmp_path)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
            except Exception as e:
                st.error(t['error_msg'].format(file_name=file.name, error=e))

        # --- Dashboard ---
        if results:
            st.markdown("---")
            st.header(t['dashboard_header'])
            df = pd.DataFrame(results)
            
            score_col = "التقييم (10)" if lang_code == 'ar' else "Score (10)"
            sentiment_col = "المشاعر" if lang_code == 'ar' else "Sentiment"
            topic_col = "الموضوع" if lang_code == 'ar' else "Topic"

            col1, col2, col3 = st.columns(3)
            col1.metric(t['metric_avg_score'], f"{df[score_col].mean():.1f}/10")
            col2.metric(t['metric_call_count'], len(df))
            col3.metric(t['metric_min_score'], df[score_col].min())

            c1, c2 = st.columns(2)
            with c1:
                st.subheader(t['chart_sentiment'])
                # إعادة رسم الرسم البياني الدائري للمشاعر
                fig_pie = px.pie(df, names=sentiment_col, color=sentiment_col, 
                             color_discrete_map={'Positive':'#4CAF50', 'Negative':'#EF5350', 'Neutral':'#FFC107'})
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                st.subheader(t['chart_topics'])
                # إعادة رسم الرسم البياني الشريطي للمواضيع
                fig_bar = px.bar(df, x=topic_col, y=score_col, color=score_col)
                st.plotly_chart(fig_bar, use_container_width=True)

            # التصدير
            def to_excel(df):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                return output.getvalue()
                
            st.download_button(t['download_btn'], data=to_excel(df), 
                               file_name='Report.xlsx', 
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')