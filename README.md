🎧 Sarcasm Detection from Voice (MUStARD++ Narrowband)

This project implements sarcasm detection based solely on audio — identifying sarcastic speech using intonation, timbre, and rhythm, not the textual content.
It processes the MUStARD++ dataset, extracting narrowband Mel-spectrograms (RGB = log-mel, ΔMFCC, ΔΔMFCC) and training a ResNet-based model using Python in Jupyter Notebook.

🧠 Overview

Traditional sarcasm detection relies on textual cues.
This notebook demonstrates that sarcasm can be detected from paralinguistic features alone — using acoustic signals that capture prosody, tone, and musicality of speech.

⚙️ Key Features

Text-free sarcasm detection — purely audio-based.

Narrowband Mel-spectrograms (optimized for 300–3400 Hz).

RGB encoding:

Red: log-mel energy

Green: ΔMFCC

Blue: ΔΔMFCC

Audio augmentations:

Gain ±6 dB

Time-stretch (0.90×, 1.10×)

Denoising (prop_decrease=0.8)

Cross-validation by group (KEY) to avoid speaker leakage.

ResNet lightweight model, trained with:

label_smoothing=0.1

batch_size=16, epochs=300

ReduceLROnPlateau, EarlyStopping, LearningRateScheduler.


Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install numpy pandas scipy librosa scikit-learn matplotlib tensorflow moviepy noisereduce tqdm pillow soundfile

```

```bash
.
├── DetectarSarcasmoDataAgumentationMustardPlus.ipynb  # main notebook
├── MUStARD_Plus_Plus-main/
│   ├── mustard_text.csv
│   └── final_utterance_videos/{KEY}.mp4
├── audio_extracted_16k/
│   └── *.wav  # audio extracted from videos (16 kHz mono)
├── out_narrowband_librosa/
│   └── narrowband_test/*.png  # RGB spectrograms
└── reports/  # metrics, confusion matrices, etc.
```

DATASET
MUStARD++: multimodal sarcasm dataset.

Extracted audio from .mp4 videos using moviepy.VideoFileClip.

Saved as 16 kHz mono WAV with codec="pcm_s16le".



Feature Extraction (Narrowband Mel)

From each WAV, features are computed using:

```bash
NARROW_WIN_MS = 80
HOP_MS = 10
n_mels = 96
```
Resulting RGB image:
```bash
R → log-mel
G → ΔMFCC
B → ΔΔMFCC
```
This encoding captures prosodic variation — crucial for sarcasm recognition.

DataAugmentation               Type	Description
Gain +6 dB / −6 dB	           Simulates microphone or environment loudness
Time-stretch 0.90× / 1.10×	   Alters speech tempo without pitch shift
Denoise	                       Removes low-level background noise

🧪 Model Training

Architecture: ResNet-Light

Optimizer: Adam()

Loss: categorical cross-entropy (label_smoothing=0.1)

Regularization: Dropout(0.5)

Early stopping and LR scheduling.

Cross-validation uses StratifiedGroupKFold (5 folds) by KEY (utterance identifier), ensuring no clip overlap between training and validation.

📊 Evaluation Metrics

Accuracy (per fold + out-of-fold mean)

Optionally: Macro-F1, ROC-AUC

Confusion matrix by fold

▶️ Inference Example

```bash
img = load_img("sample_rgb_spectrogram.png", target_size=(96,128))
x = img_to_array(img)[None, ...] / 255.0
pred = model.predict(x).argmax(1)[0]
print("Sarcastic" if pred == 1 else "Not Sarcastic")
```
⚖️ Ethical Use

Sarcasm is contextual and cultural — predictions may vary across accents or speaking styles.
Use for educational and research purposes only.
Respect dataset licenses and speaker privacy.

📄 License

MIT

