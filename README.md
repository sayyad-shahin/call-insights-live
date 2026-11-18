# Call Insights Live - real recordings
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`, add GEMINI_API_KEY (needed for automatic transcription)
3. `python app.py` and open http://localhost:5000
4. Click **Upload call**, pick a recording (WAV/MP3/M4A, max 18 MB), fill in details. Gemini transcribes it, then it is analysed.
No key? Paste a transcript (Agent:/Customer: lines) in the form and it skips transcription.
Check the transcript: if speakers look reversed, use **Swap speakers**. Evaluate model: `python evaluate.py`
"# call-insights-live" 
