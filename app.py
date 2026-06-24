import streamlit as st
import tempfile
import pandas as pd
import pickle
import os
from movement_analysis import analyze_video

st.set_page_config(
    page_title="Pediatric Movement Analysis",
    layout="wide"
)

st.title("🎬 AI-Based Pediatric Movement Analysis")

st.write("""
Upload a video and generate movement metrics with anxiety prediction.
""")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:

    st.video(uploaded_file)

    if st.button("🔍 Analyze Video"):

        with st.spinner("Analyzing..."):

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            output_csv = analyze_video(temp_path)

        st.success("✅ Analysis Complete")

        # Read and display the movement metrics
        results_df = pd.read_csv(output_csv)
        
        # Display metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Movement Metrics")
            st.metric("Duration (minutes)", f"{results_df['Duration_Minutes'].values[0]:.2f}")
            st.metric("Eye Blinks", int(results_df['Eye_Blinks'].values[0]))
            st.metric("Blink Rate (per min)", f"{results_df['Blink_Rate_Per_Minute'].values[0]:.2f}")
        
        with col2:
            st.subheader("🎯 Movement Counts")
            st.metric("Head Movements", int(results_df['Head_Movements'].values[0]))
            # Hand movements removed from analysis; skip display if absent
            if 'Hand_Movements' in results_df.columns:
                st.metric("Hand Movements", int(results_df['Hand_Movements'].values[0]))
            st.metric("Body Movements", int(results_df['Body_Movements'].values[0]))
        
        # Display full results table
        st.subheader("📈 Detailed Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Anxiety prediction using the trained model
        try:
            if os.path.exists("anxiety_model.pkl"):
                with open("anxiety_model.pkl", "rb") as f:
                    model = pickle.load(f)
                
                # Prepare features for prediction; ensure required columns exist
                candidate_features = [
                    "Eye_Blinks",
                    "Head_Movements",
                    "Hand_Movements",
                    "Body_Movements"
                ]
                available_features = [c for c in candidate_features if c in results_df.columns]

                if not available_features:
                    st.warning("Not enough features available for anxiety prediction.")
                else:
                    features = results_df[available_features].values

                    # Ensure model input dimension matches; otherwise skip prediction
                    try:
                        expected = getattr(model, 'n_features_in_', None)
                        if expected is not None and expected != features.shape[1]:
                            st.warning(f"Model expects {expected} features but {features.shape[1]} available; skipping prediction.")
                        else:
                            anxiety_prediction = model.predict(features)[0]
                            prediction_proba = model.predict_proba(features)[0]
                            st.subheader("🧠 Anxiety Prediction")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Predicted Anxiety Level", f"{anxiety_prediction:.2f}")
                            if len(prediction_proba) > 1:
                                with col2:
                                    st.metric("Confidence", f"{max(prediction_proba)*100:.1f}%")
                            if anxiety_prediction < 0.33:
                                st.success("✅ Low Anxiety Level")
                            elif anxiety_prediction < 0.67:
                                st.warning("⚠️ Moderate Anxiety Level")
                            else:
                                st.error("🔴 High Anxiety Level")
                    except Exception as e:
                        st.warning(f"Prediction failed: {e}")
                
                st.subheader("🧠 Anxiety Prediction")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Anxiety Level", f"{anxiety_prediction:.2f}")
                
                # Display confidence scores if available
                if len(prediction_proba) > 1:
                    with col2:
                        st.metric("Confidence", f"{max(prediction_proba)*100:.1f}%")
                
                # Color-coded prediction
                if anxiety_prediction < 0.33:
                    st.success("✅ Low Anxiety Level")
                elif anxiety_prediction < 0.67:
                    st.warning("⚠️ Moderate Anxiety Level")
                else:
                    st.error("🔴 High Anxiety Level")
        
        except Exception as e:
            st.warning(f"Could not load anxiety model: {e}")
        
        # Download button
        with open(output_csv, "rb") as file:
            st.download_button(
                label="📥 Download CSV",
                data=file,
                file_name="movement_metrics.csv",
                mime="text/csv"
            )