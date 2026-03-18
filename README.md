# Sign Language Translator - User Guide

This project allows you to collect hand gesture data, train a model, and use it for real-time sign language translation.

## Prerequisites

Ensure you have installed the requirements:
```bash
pip install -r requirements.txt
```

---

## ⌨️ Controls & Shortcuts Reference

| Action | Script / Tool | Key / Trigger |
| :--- | :--- | :--- |
| **Start / Stop Recording** | `data_collection.py` | Press **'r'** |
| **Quit & Close Camera window** | `data_collection.py` / `inference.py` | Press **'q'** |
| **Open Dashboard GUI** | (Terminal) | `streamlit run dashboard.py` |

---

## Step 1: Data Collection

You need to record landmarks for each sign/letter you want to translate.

Run the script specifying the **label** you are recording:

```bash
python data_collection.py --label A
```

**Inside the script:**
-   Ensure your hand is visible in the webcam.
-   Press **'r'** to **START** recording. Move your hand slightly to capture different angles/distances.
-   Press **'r'** again to **STOP** recording.
-   The count of frames will increase. Collect around **100-200 frames** per label for best results.
-   Press **'q'** to exit.

Repeat this for other letters/signs:
```bash
python data_collection.py --label B
python data_collection.py --label C
```

All data will be appended to `gestures.csv`.

---

## Step 2: Model Training

Once you have collected data for a few signs, train the classification model.

Run:
```bash
python train_model.py
```

This will:
1.  Load `gestures.csv`.
2.  Train a Random Forest classifier.
3.  Print accuracy.
4.  Save `model.pkl` and `label_encoder.pkl`.

---

## Step 3: Real-Time Translation

Test your model live on the webcam.

Run:
```bash
python inference.py
```

-   The window will show the predicted label if confidence is high.
-   Press **'q'** to exit.

---

---

## 🖐️ How to Perform Signs (ASL Reference)

To test your translator effectively, refer to the physical hand shapes below (based on American Sign Language):

| Letter | How to Shape Your Hand |
| :--- | :--- |
| **A** | Make a **fist**. Leave your thumb resting on the side of index finger pointing up. |
| **B** | Hold your hand **flat, fingers together** pointing up. Tuck your thumb into your palm. |
| **C** | Curve your fingers and thumb together into a **cup shape** (like holding a ball). |
| **D** | Lift your **Index finger straight up**. Curl the other 3 fingers to touch your thumb (making a circle). |
| **E** | Curl all 4 fingers down to touch the top of your thumb (like a **claw**). |
| **F** | Touch your **Index and Thumb tips together** (forming an "OK" sign). Lift other 3 fingers up. |
| **G** | Point your Index and Thumb forward (like **pinching** something). |
| **H** | Point Index and Middle fingers forward together (flat side). |
| **I** | Lift your **Little finger (pinky) straight up**, curl other fingers into a fist. |
| **J** | Standard **I** shape + draw a "J" curve in the air with your pinky. |
| **K** | Index and Middle up making a **"V"**, Thumb pressing against middle joint. |
| **L** | Index finger straight up, Thumb pointing out making an **"L"** shape. |
| **M** | Fist with Thumb tucked UNDER Index, Middle, and Ring finger completely. |
| **N** | Fist with Thumb tucked UNDER Index and Middle finger ONLY. |
| **O** | Curl all fingers and thumb together to make an **"O"** circle shape. |
| **P** | Same as **K** but point your Index and Middle fingers **downwards**. |
| **Q** | Same as **G** but point your Index and Thumb **downwards**. |
| **R** | Cross your **Index and Middle fingers** together pointing straight up. |
| **S** | Make a solid **fist**, resting your Thumb across the front of all fingers. |
| **T** | Fist with Thumb tucked under the **Index finger only**. |
| **U** | Index and Middle fingers together pointing straight up. |
| **V** | Index and Middle pointing up separated in a **"V"** (Peace sign). |
| **W** | Index, Middle, and Ring fingers pointing up together (spread slightly). |
| **X** | Curl your Index finger into a **hook** pointing up, other is in a fist. |
| **Y** | Extend only your **Pinky and Thumb** out/up (like a phone sign). |
| **Z** | Lift Index finger and trace a **"Z"** path in the air. |
| **Hello** | Place hand flat against temple, moving away from head (**Salute**). |
| **How are You?** | Touch backs of fingers together at chest and rotate forward/down. |
| **I am Fine** | Open hand with spread fingers, touch **Thumb tip into center of chest**. |

> 📚 **For a full A-Z visual chart, you can look up "ASL Alphabet Chart" on Google Images for assistance.**

---

## 💡 Tips for Better Accuracy
-   Collect data in same lighting conditions as you will test in.
-   Vary hand position (close, far, slightly rotated) during data collection.
-   Ensure only **ONE** hand is in frame for best performance.

---

```bash
streamlit run dashboard.py
```
