# Boxing Highlight Detection
ML-powered boxing highlight detector — combines video motion and audio energy analysis with Scikit-learn to automatically identifyfight highlights

This project uses computer vision, audio signal processing, and ML to automatically detect and extract highlight moments (knockdowns, heavy exchanges, or KOs) from full boxing match videos.
The system analyzes both visual motion patterns and crowd / commentatory reaction audio cues to identify these moments.

Currently outputs timestamps but I am working an getting an extracted video of clipped highlight moments as the output.


- Upload a boxing match video
- Extract frames and audio
- Compute motion and audio energy features
- Predict highlight probability per second via trained sckikit-learn model
- Testing and results via streamlit dashboard

 Currently working towards integrating deep learning with PyTorch to enhance the highlight generating pipeline.  
 
 ## Run the App
 python -m streamlit run streamlit_app/app.py

Note: To use FFmpeg for audio extration it must be installed and accessible via the $PATH environment variable. Without this audio extraction will fall back to moviepy - less robust but still functional for most inputs
There are a variety of ways to install FFmpeg, such as the official download links, or using your package manager of choice (e.g. sudo apt install ffmpeg on Debian/Ubuntu, brew install ffmpeg on OS X, etc.).
See https://github.com/kkroening/ffmpeg-python for more information
