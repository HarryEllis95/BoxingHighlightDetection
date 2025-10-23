# BoxingHighlightDetection
AI-powered boxing highlight detector — combines video motion and audio energy analysis with PyTorch deep learning to automatically identify and clip fight highlights

This project uses computer vision, audio signal processing, and deep learning to automatically detect and extract highlight moments (knockdowns, heavy exchanges, or KOs) from full boxing match videos.
The system analyzes both visual motion patterns and crowd reaction audio cues to identify these moments.

Currently outputs timestamps but I am working an getting an extracted video of clipped highlight moments as the output.


- Download a boxing match video using yt-dlp
- Extract frames and audio
- Compute motion and audio energy features
- Predict highlight probability per second via trained PyTorch model
