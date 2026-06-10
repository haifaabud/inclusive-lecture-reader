# Inclusive Lecture Reader

Real-time assistive lecture assistant MVP for visually impaired students.

## Run
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open:
   ```text
   http://127.0.0.1:5000
   ```

## Notes
- The app uses the computer's webcam through OpenCV.
- The browser uses speech synthesis for audio output.
- If the camera is unavailable, use the image upload fallback on the page.
