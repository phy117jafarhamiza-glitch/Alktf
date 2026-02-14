import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.set_page_config(page_title="Snatch Phase Analyzer", layout="wide")
st.title("🏋️ محلل مراحل رفعة الخطف (الاستعداد ثم السحب)")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

video_file = st.file_uploader("ارفع فيديو الرفعة الجانبي", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    error_flags = {"hip_high": False, "hip_low": False, "early_back": False, "bar_away": False}
    feedbacks = []
    
    # متغيرات لتحديد لحظة بدء الحركة
    movement_started = False
    initial_wrist_y = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            sh_y, hip_y, knee_y = lm[12].y, lm[24].y, lm[26].y
            wrist_x, wrist_y = lm[16].x, lm[16].y
            ankle_x = lm[28].x

            # --- الخطوة 1: اكتشاف لحظة بدء الحركة ---
            if initial_wrist_y is None:
                initial_wrist_y = wrist_y # تخزين موقع البار في أول إطار

            # إذا تحرك المعصم للأعلى بمسافة ملحوظة، نعلن بدء السحبة الأولى
            if not movement_started and abs(wrist_y - initial_wrist_y) > 0.02:
                movement_started = True

            # --- الخطوة 2: تحليل وقفة الاستعداد (قبل الحركة فقط) ---
            if not movement_started:
                if not (error_flags["hip_high"] or error_flags["hip_low"]):
                    if hip_y < sh_y + 0.05:
                        error_flags["hip_high"] = True
                        feedbacks.append("❌ الاستعداد: الحوض مرتفع جداً قبل بدء السحب.")
                    elif hip_y > knee_y - 0.05:
                        error_flags["hip_low"] = True
                        feedbacks.append("❌ الاستعداد: الحوض منخفض جداً (وضعية قرفصاء وليست استعداد).")
                
                cv2.putText(frame, "PHASE: SETUP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # --- الخطوة 3: تحليل السحبة الأولى (بعد بدء الحركة) ---
            else:
                cv2.putText(frame, "PHASE: FIRST PULL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if not error_flags["early_back"]:
                    # قياس إذا كان الحوض يرتفع أسرع من الكتف في بداية السحب
                    if hip_y > sh_y + 0.2: 
                        error_flags["early_back"] = True
                        feedbacks.append("❌ السحبة الأولى: تقوس الظهر (ارتفاع الحوض أسرع من الصدر).")

            # تحليل المسار (مستمر طوال الحركة)
            if movement_started and not error_flags["bar_away"]:
                if abs(wrist_x - ankle_x) > 0.18:
                    error_flags["bar_away"] = True
                    feedbacks.append("❌ المسار: البار يبتعد عن مسار القدمين.")

        st_frame.image(frame, channels="BGR", use_column_width=True)
    cap.release()

    # --- النتائج النهائية والدرجات ---
    score_setup = 5 if not (error_flags["hip_high"] or error_flags["hip_low"]) else 2
    score_pull = 3 if not error_flags["early_back"] else 1
    score_catch = 5 if not error_flags["bar_away"] else 3
    total_score = score_setup + score_pull + score_catch + 2

    st.divider()
    st.header(f"النتيجة: {total_score} / 15")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 تقييم المراحل")
        st.write(f"1️⃣ وقفة الاستعداد: {score_setup}/5")
        st.write(f"2️⃣ السحبة الأولى: {score_pull}/3")
        st.write(f"3️⃣ السقوط والثبات: {score_catch + 2}/7")
        
    with col2:
        st.subheader("💡 تحليل الأخطاء والنصائح")
        if feedbacks:
            for error in feedbacks: st.error(error)
        else: st.success("ممتاز! حافظت على الفصل الصحيح بين مراحل الرفعة.")
