import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.set_page_config(page_title="Snatch Visual Coach", layout="wide")
st.title("🏋️ مدرب الخطف: تصحيح الأخطاء بالرسم التوضيحي")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

video_file = st.file_uploader("ارفع فيديو الرفعة الجانبي", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    error_flags = {"hip_high": False, "hip_low": False, "early_back": False}
    initial_wrist_y = None
    movement_started = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # تحويل الإحداثيات لنقاط بكسلية للرسم
            def get_pix(landmark_idx):
                return int(lm[landmark_idx].x * w), int(lm[landmark_idx].y * h)

            sh_p = get_pix(12) # الكتف
            hip_p = get_pix(24) # الحوض
            knee_p = get_pix(26) # الركبة
            wrist_p = get_pix(16) # المعصم (البار)

            # --- كشف بدء الحركة ---
            if initial_wrist_y is None: initial_wrist_y = lm[16].y
            if not movement_started and abs(lm[16].y - initial_wrist_y) > 0.02:
                movement_started = True

            # --- الرسم التوضيحي لتصحيح الأخطاء ---
            
            # 1. تصحيح وقفة الاستعداد (قبل الحركة)
            if not movement_started:
                # إذا كان الحوض مرتفعاً جداً (قريب من مستوى الكتف)
                if lm[24].y < lm[12].y + 0.05:
                    error_flags["hip_high"] = True
                    # رسم سهم توضيحي لخفض الحوض
                    cv2.arrowedLine(frame, hip_p, (hip_p[0], hip_p[1] + 50), (0, 0, 255), 5)
                    cv2.putText(frame, "LOWER YOUR HIP", (hip_p[0]+10, hip_p[1]+30), 1, 1.5, (0,0,255), 2)
                
                # إذا كان الحوض منخفضاً جداً
                elif lm[24].y > lm[26].y - 0.05:
                    error_flags["hip_low"] = True
                    # رسم سهم توضيحي لرفع الحوض
                    cv2.arrowedLine(frame, hip_p, (hip_p[0], hip_p[1] - 50), (0, 0, 255), 5)
                    cv2.putText(frame, "RAISE YOUR HIP", (hip_p[0]+10, hip_p[1]-30), 1, 1.5, (0,0,255), 2)

            # 2. تصحيح السحبة الأولى (بعد الحركة)
            else:
                if lm[24].y > lm[12].y + 0.2: # الصدر يسقط والحوض يرتفع
                    error_flags["early_back"] = True
                    # رسم خط توضيحي للصدر ليظهر أنه يجب أن يرتفع
                    cv2.line(frame, sh_p, (sh_p[0], sh_p[1]-60), (0, 255, 255), 5)
                    cv2.putText(frame, "KEEP CHEST UP", (sh_p[0]-50, sh_p[1]-70), 1, 1.5, (0,255,255), 2)

            # رسم الهيكل العظمي الأساسي
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        st_frame.image(frame, channels="BGR", use_column_width=True)
    cap.release()

    st.success("انتهى التحليل المكتبي. راجع الرسومات الحمراء والصفراء على الفيديو لتصحيح وضعيتك.")
