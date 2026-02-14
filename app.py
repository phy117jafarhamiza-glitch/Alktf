import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# إعدادات الصفحة
st.set_page_config(page_title="Snatch Technical Evaluator", layout="wide")
st.title("🏋️ تقييم الأداء الفني لرفعة الخطف (من 15 درجة)")

# إعداد MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

video_file = st.file_uploader("ارفع فيديو الرفعة للتقييم", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    
    # متغيرات التقييم
    scores = {"setup": 0, "first_pull": 0, "catch": 0, "stability": 0}
    max_velocity = 0
    min_back_angle = 180
    
    path_points = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # استخراج النقاط
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # 1. تقييم وقفة الاستعداد (5 درجات)
            back_angle = calculate_angle(shoulder, hip, knee)
            if 40 < back_angle < 70: scores["setup"] = 5
            elif 30 < back_angle < 80: scores["setup"] = 3
            
            # 2. تقييم السحبة الأولى (3 درجات)
            # نراقب ثبات زاوية الظهر أثناء الصعود
            if back_angle < min_back_angle: min_back_angle = back_angle
            if abs(back_angle - min_back_angle) < 10: scores["first_pull"] = 3
            else: scores["first_pull"] = 1

            # 3. السقوط تحت الثقل (5 درجات)
            # يقاس بعمق الحوض بالنسبة للركبة
            if hip[1] > knee[1]: scores["catch"] = 5
            elif hip[1] > knee[1] - 0.1: scores["catch"] = 3

            # 4. الوقوف والثبات (2 درجة)
            if abs(wrist[0] - hip[0]) < 0.1: scores["stability"] = 2

            # رسم المسار
            cx, cy = int(wrist[0] * w), int(wrist[1] * h)
            path_points.append((cx, cy))
            for i in range(1, len(path_points)):
                cv2.line(frame, path_points[i-1], path_points[i], (0, 255, 0), 2)

        st_frame.image(frame, channels="BGR", use_column_width=True)

    cap.release()

    # عرض النتيجة النهائية
    total_score = sum(scores.values())
    st.header(f"التقييم النهائي: {total_score} / 15")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"🔹 وقفة الاستعداد: {scores['setup']} / 5")
        st.write(f"🔹 السحبة الأولى: {scores['first_pull']} / 3")
    with col2:
        st.write(f"🔹 السقوط تحت الثقل: {scores['catch']} / 5")
        st.write(f"🔹 الوقوف والثبات: {scores['stability']} / 2")

    if total_score >= 12: st.balloons()
