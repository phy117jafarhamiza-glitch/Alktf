import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# إعدادات واجهة التطبيق
st.set_page_config(page_title="Snatch Pro Evaluator", layout="wide")

# رسالة تنبيهية تظهر في البداية
st.warning("⚠️ ملاحظة هامة: للحصول على تقييم دقيق، يجب أن يكون التصوير من الجانب (Side View) وبشكل أفقي تماماً.")
st.title("🏋️ نظام تقييم رفعة الخطف والتحليل الفني")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

video_file = st.file_uploader("قم برفع فيديو الرفعة الجانبي هنا", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    
    # متغيرات التقييم والملاحظات
    scores = {"setup": 5, "first_pull": 3, "catch": 5, "stability": 2}
    feedbacks = []
    max_path_deviation = 0
    path_points = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # تحديد النقاط الرئيسية
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # --- تحليل الأخطاء وتحديد مكانها ---
            
            # 1. خطأ وقفة الاستعداد
            back_angle = calculate_angle(shoulder, hip, knee)
            if back_angle < 35 or back_angle > 75:
                if "وضع الحوض خاطئ في البداية" not in feedbacks:
                    feedbacks.append("وضع الحوض خاطئ في البداية (منخفض جداً أو مرتفع جداً)")
                    scores["setup"] -= 2

            # 2. خطأ السحبة الأولى (تقوس الظهر)
            if back_angle > 85 and wrist[1] > knee[1]:
                if "رفع الظهر مبكراً" not in feedbacks:
                    feedbacks.append("خطأ في السحبة الأولى: قمت برفع الظهر قبل عبور البار للركبة")
                    scores["first_pull"] -= 1

            # 3. خطأ مسار البار (الابتعاد عن الجسم)
            if len(path_points) > 0:
                deviation = abs(wrist[0] - ankle[0])
                if deviation > 0.15: # إذا ابتعد البار عن خط الكاحل بمسافة كبيرة
                    cv2.putText(frame, "BAR DISTANCE ERROR!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    if "البار بعيد عن الجسم" not in feedbacks:
                        feedbacks.append("المسار الفني: البار يبتعد عن جسمك بشكل كبير (Looping)")
                        scores["catch"] -= 1

            # رسم المسار والنقاط
            cx, cy = int(wrist[0] * w), int(wrist[1] * h)
            path_points.append((cx, cy))
            for i in range(1, len(path_points)):
                cv2.line(frame, path_points[i-1], path_points[i], (0, 255, 0), 2)

        st_frame.image(frame, channels="BGR", use_column_width=True)

    cap.release()

    # --- عرض لوحة النتائج النهائية ---
    st.divider()
    total_score = sum(scores.values())
    st.header(f"النتيجة النهائية: {total_score} / 15")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 تفاصيل الدرجات")
        st.write(f"✅ الاستعداد: {scores['setup']}/5")
        st.write(f"✅ السحبة الأولى: {scores['first_pull']}/3")
        st.write(f"✅ السقوط: {scores['catch']}/5")
        st.write(f"✅ الثبات: {scores['stability']}/2")
        
    with col2:
        st.subheader("❌ الأخطاء المكتشفة")
        if feedbacks:
            for error in feedbacks:
                st.error(error)
        else:
            st.success("أداء ممتاز! لم يتم رصد أخطاء فنية كبرى.")
