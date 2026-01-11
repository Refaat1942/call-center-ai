import streamlit as st
import openai
import os
import tempfile
import pandas as pd
import plotly.express as px
import json
import io

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Lotus Professional QA System",
    page_icon="🎧",
    layout="wide"
)

# --- معايير التقييم (مستخرجة من الصور) ---
QA_CRITERIA = {
    "Non-Critical": [
        "Greeting", "Voice Tone", "Using Customer's Name", 
        "Active Listening & Interruption", "Using Professional Language", 
        "Hold & Transfer Processes", "Mute/Dead Air", "Closing",
        "Collecting and Verifying Data"
    ],
    "End User Critical": [
        "Entering Collected Data Correctly", "Entering Transaction Correctly",
        "Providing Accurate Information", "Inappropriate/Rude Behavior",
        "Controlling the Call", "Documenting Call Details"
    ],
    "Compliance Critical": [
        "Sharing Customer Data with Other Party"
    ]
}

# --- النصوص والترجمة ---
translations = {
    'ar': {
        'title': 'نظام لوتس لتحليل جودة المكالمات (QA Automation)',
        'upload_label': 'رفع تسجيلات المكالمات',
        'sidebar': 'الإعدادات',
        'start_btn': 'ابدأ التحليل الذكي لـ {count} مكالمات',
        'analyzing': 'جاري معالجة: {file}... (فصل المتحدثين + تقييم المعايير)',
        'result_header': 'نتائج التقييم',
        'critical_alert': '⚠️ خطأ قاتل (Critical Error)',
        'score': 'النتيجة النهائية',
        'agent': 'الموظف',
        'customer': 'العميل',
        'download': 'تحميل تقرير Excel شامل',
        'pass': 'مطابق',
        'fail': 'غير مطابق',
        'na': 'غير منطبق'
    }
}
t = translations['ar']

# --- التنسيق (CSS) ---
st.markdown("""
<style>
    .stApp {background-color: #f0f2f6;}
    .pass-badge {background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-weight: bold;}
    .fail-badge {background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; font-weight: bold;}
    .critical-fail {border: 2px solid red; background-color: #ffe6e6; padding: 10px; border-radius: 5px;}
    .metric-box {background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;}
</style>
""", unsafe_allow_html=True)

# --- الشريط الجانبي ---
with st.sidebar:
    st.image("image_5.png", use_container_width=True) # تأكد من وجود الصورة
    st.title(t['sidebar'])
    
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        
    st.info("💡 هذا النظام يستخدم GPT-4o لفصل المتحدثين بدقة وتطبيق معايير الـ QA الخاصة بالشركة.")

# --- دوال الذكاء الاصطناعي ---

def transcribe_audio(client, audio_path):
    """تحويل الصوت لنص خام"""
    with open(audio_path, "rb") as audio_file:
        return client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text"
        )

def format_dialogue(client, raw_text):
    """مرحلة 1: تحويل النص الخام إلى حوار منظم (Agent vs Customer)"""
    prompt = f"""
    You are a transcript formatter. Convert the following raw Arabic text into a structured dialogue script.
    Identify the "Agent" (Call Center Employee) and the "Customer" based on context (e.g., who says 'Hello, this is [Name] from [Company]').
    
    Format:
    Agent: [Text]
    Customer: [Text]
    
    Raw Text:
    {raw_text}
    """
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

def analyze_qa_criteria(client, dialogue):
    """مرحلة 2: تقييم الجودة بناءً على المعايير المحددة"""
    
    criteria_json = json.dumps(QA_CRITERIA, ensure_ascii=False)
    
    prompt = f"""
    Act as a strict Quality Assurance (QA) Specialist. Evaluate the following Call Center dialogue based on the provided Criteria List.
    
    **Dialogue:**
    {dialogue}
    
    **Criteria List:**
    {criteria_json}
    
    **Instructions:**
    1. For EACH item in the criteria list, determine if it is "PASS", "FAIL", or "N/A" (Not Applicable).
    2. Provide a short "reason" for the evaluation (in Arabic).
    3. Calculate a "Final Score" out of 100.
       - Start with 100.
       - Deduct 5 points for each "Non-Critical" FAIL.
       - Deduct 100 points (Zero out) for ANY "Critical" FAIL (End User or Compliance).
    4. Provide a brief Arabic summary of the call.
    
    **Output JSON Format (Strictly):**
    {{
        "final_score": 85,
        "critical_error_found": false,
        "summary": "ملخص المكالمة...",
        "details": [
            {{"category": "Non-Critical", "item": "Greeting", "status": "PASS", "reason": "بدأ بالتحية القياسية"}},
            {{"category": "End User Critical", "item": "Providing Accurate Information", "status": "FAIL", "reason": "أعطى معلومة خاطئة عن الفاتورة"}}
        ]
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a JSON output machine."}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)

# --- الواجهة الرئيسية ---
st.title(t['title'])

if not api_key:
    st.warning("⚠️ يرجى إدخال مفتاح API للمتابعة.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

uploaded_files = st.file_uploader(t['upload_label'], type=['mp3', 'wav', 'm4a'], accept_multiple_files=True)

if uploaded_files and st.button(t['start_btn'].format(count=len(uploaded_files))):
    
    full_report_data = []
    
    for file in uploaded_files:
        try:
            with st.spinner(t['analyzing'].format(file=file.name)):
                # 1. حفظ الملف مؤقتاً
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                # 2. التحويل النصي
                raw_text = transcribe_audio(client, tmp_path)
                
                # 3. تنظيم الحوار (فصل المتحدثين)
                structured_dialogue = format_dialogue(client, raw_text)
                
                # 4. تحليل الجودة
                qa_result = analyze_qa_criteria(client, structured_dialogue)
                
                # تنظيف
                os.remove(tmp_path)
                
                # عرض النتائج لهذه المكالمة
                score_color = "red" if qa_result['final_score'] < 70 else "green"
                
                with st.expander(f"📞 {file.name} | النتيجة: :{score_color}[{qa_result['final_score']}%]"):
                    
                    # تنبيه الأخطاء القاتلة
                    if qa_result.get('critical_error_found'):
                        st.error(f"🚨 {t['critical_alert']} - تم تصفير النتيجة!")
                    
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        st.markdown(f"**الملخص:** {qa_result['summary']}")
                        st.markdown("**تفاصيل التقييم:**")
                        
                        # إنشاء جدول للنتائج
                        details_df = pd.DataFrame(qa_result['details'])
                        
                        # دالة لتلوين الخلايا
                        def color_status(val):
                            color = '#d4edda' if val == 'PASS' else '#f8d7da' if val == 'FAIL' else '#fff3cd'
                            return f'background-color: {color}; color: black; font-weight: bold;'
                        
                        st.dataframe(details_df.style.applymap(color_status, subset=['status']), use_container_width=True)

                    with c2:
                        st.markdown("**📝 سجل المكالمة (Agent vs Customer):**")
                        st.text_area("نص المكالمة", structured_dialogue, height=400)
                
                # تجميع البيانات للتقرير النهائي
                flat_record = {
                    "File Name": file.name,
                    "Final Score": qa_result['final_score'],
                    "Critical Error": "YES" if qa_result.get('critical_error_found') else "NO",
                    "Summary": qa_result['summary']
                }
                # إضافة تفاصيل البنود كأعمدة
                for item in qa_result['details']:
                    flat_record[item['item']] = item['status']
                    
                full_report_data.append(flat_record)

        except Exception as e:
            st.error(f"حدث خطأ في الملف {file.name}: {str(e)}")

    # --- لوحة القيادة النهائية ---
    if full_report_data:
        st.markdown("---")
        st.header("📊 التحليل المجمع (Dashboard)")
        
        df_report = pd.DataFrame(full_report_data)
        
        # مؤشرات الأداء
        m1, m2, m3 = st.columns(3)
        avg_score = df_report['Final Score'].mean()
        m1.metric("متوسط الجودة", f"{avg_score:.1f}%")
        m2.metric("عدد المكالمات", len(df_report))
        fatal_count = len(df_report[df_report['Critical Error'] == 'YES'])
        m3.metric("مكالمات بها أخطاء قاتلة", fatal_count, delta_color="inverse")
        
        # رسم بياني للأخطاء
        st.subheader("توزيع نتائج البنود (Pass vs Fail)")
        
        # تحويل البيانات للرسم
        long_df = pd.melt(df_report, id_vars=['File Name', 'Final Score', 'Critical Error', 'Summary'], 
                          var_name='Criteria', value_name='Status')
        
        fig = px.histogram(long_df, x='Criteria', color='Status', 
                           color_discrete_map={'PASS': '#28a745', 'FAIL': '#dc3545', 'N/A': '#6c757d'},
                           barmode='group')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # زر التحميل
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_report.to_excel(writer, index=False)
        
        st.download_button(
            label=t['download'],
            data=output.getvalue(),
            file_name='Lotus_QA_Report.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )