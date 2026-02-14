import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.set_page_config(page_title="Snatch Exam Pro", layout="wide")
st.title("🏋️ نظام تقييم رفعة الخطف المتكامل (15 درجة)")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

video_file = st.file_uploader("ارفع فيديو أداء الطالب (تصوير جانبي فقط)", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    # مصفوفة الأخطاء والأعلام
    error_flags = {"hip_high": False, "hip_low": False, "early_back": False, "shallow_catch": False, "unstable": False}
    error_images = {}
    movement_started = False
    initial_wrist_y = None
    max_catch_depth = 0 # لتتبع أقصى نزول للحوض

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def get_p(idx): return int(lm[idx].x * w), int(lm[idx].y * h)
            
            sh, hip, knee, wrist, ankle = get_p(12), get_p(24), get_p(26), get_p(16), get_p(28)

            # 1. كشف بدء الحركة
            if initial_wrist_y is None: initial_wrist_y = lm[16].y
            if not movement_started and abs(lm[16].y - initial_wrist_y) > 0.03:
                movement_started = True

            # --- أ. وقفة الاستعداد (5 درجات) ---
            if not movement_started:
                if lm[24].y < lm[12].y + 0.05 and not error_flags["hip_high"]:
                    error_flags["hip_high"] = True
                    img_err = frame.copy()
                    cv2.arrowedLine(img_err, hip, (hip[0], hip[1] + 60), (0, 0, 255), 6)
                    error_images["الاستعداد"] = {"img": img_err, "tip": "الحوض مرتفع جداً؛ اخفض الحوض لتبدأ السحب بقوة الساقين."}
                elif lm[24].y > lm[26].y - 0.05 and not error_flags["hip_low"]:
                    error_flags["hip_low"] = True
                    img_err = frame.copy()
                    cv2.arrowedLine(img_err, hip, (hip[0], hip[1] - 60), (0, 0, 255), 6)
                    error_images["الاستعداد"] = {"img": img_err, "tip": "الحوض منخفض جداً؛ ارفعه قليلاً لتجنب وضعية القرفصاء."}

            # --- ب. السحبة الأولى (3 درجات) ---
            elif movement_started and wrist[1] > knee[1]:
                if lm[24].y > lm[12].y + 0.22 and not error_flags["early_back"]:
                    error_flags["early_back"] = True
                    img_err = frame.copy()
                    cv2.line(img_err, sh, (sh[0], sh[1]-80), (0, 255, 255), 6)
                    error_images["السحبة الأولى"] = {"img": img_err, "tip": "حافظ على صدرك مرتفعاً؛ الحوض يرتفع أسرع من اللازم."}

            # --- ج. السقوط تحت الثقل (5 درجات) ---
            if movement_started:
                # تتبع أقصى عمق للحوض
                if lm[24].y > max_catch_depth: max_catch_depth = lm[24].y
                
                # إذا انتهت الرفعة ولم ينزل الحوض أسفل الركبة
                if max_catch_depth < lm[26].y and wrist[1] < sh[1]:
                    error_flags["shallow_catch"] = True

            # --- د. الوقوف والثبات (2 درجة) ---
            # قياس المسافة الأفقية بين المعصم والكعب عند نهاية الرفعة
            if movement_started and wrist[1] < sh[1]: # المعصم فوق الرأس
                if abs(lm[16].x - lm[28].x) > 0.15:
                    error_flags["unstable"] = True

        st_frame.image(frame, channels="BGR", use_column_width=True)
    cap.release()

    # --- حساب الدرجات النهائي ---
    s_setup = 2 if (error_flags["hip_high"] or error_flags["hip_low"]) else 5
    s_pull = 1 if error_flags["early_back"] else 3
    s_catch = 2 if error_flags["shallow_catch"] else 5
    s_stable = 0 if error_flags["unstable"] else 2
    total = s_setup + s_pull + s_catch + s_stable

    # عرض النتائج
    st.divider()
    st.header(f"النتيجة النهائية للطالب: {total} / 15")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("الاستعداد", f"{s_setup}/5")
    c2.metric("السحبة 1", f"{s_pull}/3")
    c3.metric("السقوط", f"{s_catch}/5")
    c4.metric("الثبات", f"{s_stable}/2")

    if error_images:
        st.subheader("📸 تحليل الأخطاء البصري")
        for key, data in error_images.items():
            col_a, col_b = st.columns([1, 1])
            with col_a: st.image(data["img"], channels="BGR")
            with col_b: st.error(f"خطأ في {key}"); st.info(data["tip"])
