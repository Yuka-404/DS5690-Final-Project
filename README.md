# Reality 2.0: Real-Time Video Object Editing (SAM 2)

Presenter: Kunyang Ji

Date: 2025.12.4

# 1. Overview:

**The Problem:** Computer vision has mastered static images, but video remains a challenge due to "Temporal Inconsistency." When editing video, if you identify an object (like a car) in Frame 1, current models often "lose" it by Frame 50 if it rotates, changes lighting, or goes behind a tree (occlusion). This makes automated video editing (VFX) impossible without manual frame-by-frame work.

**Proposed Approach:** This project implements Meta’s Segment Anything Model 2 (SAM 2). Unlike previous models that treat video as 3D blocks, SAM 2 treats video as a stream of images with a "Memory Bank." It allows users to click an object once, and the model propagates that selection through the entire video in real-time.

![Promptable Visual Segmentation (PVS) Task](./images/fig7.png)

This graph illustrates the Promptable Visual Segmentation (PVS) task, showing how it generalizes both "Segment Anything" (images) and "VOS" (video) into a unified task.

# 2. Methdology:

**2.1. Theory**: The "Memory Attention" Mechanism 

We apply Transformer Attention mechanisms to the temporal dimension.

- Standard Transformers: Attend to all words in a sentence.

- SAM 2: Implements "Streaming Memory Attention." It maintains a "Memory Bank" (FIFO Queue) of the object’s embeddings from past frames. When processing the current frame, the model attends to these past embeddings to resolve ambiguity.

**2.2. Architecture Pipeline We utilize a unified pipeline consisting of three components:**


**Image Encoder (Hiera):** A hierarchical Vision Transformer that runs at ~44 FPS, converting video frames into feature maps.

**Memory Encoder:** Compresses the current mask prediction into a lightweight "memory token" and stores it.

**Mask Decoder:** Fuses the Current Frame + Memory Context + User Clicks to generate the final mask.

![Mask Decoder Architecture](./images/fig8.png)

This is the internal "brain" of the model that generates the masks. It closely follows the Transformer architecture mentioned in your syllabus.

**Attention Mechanisms:** It uses Self-Attention (relating prompts to each other) and Cross-Attention (labeled "token to image attn" and "image to token attn") to mix the user's prompt with the image features.

**New Video Features:** Unlike the original SAM, this decoder has an "occlusion head" (bottom right). It predicts an occlusion score to tell the system if the object is currently hidden behind something else (e.g., a car driving behind a tree).


**2.3. Dataset-Driven Validation:** 
To ensure reliability, we validate this methodology using the SA-V (Segment Anything Video) dataset logic. We test the model's ability to handle "Disappearance/Reappearance" events, which occur in 42.5% of SA-V tracks, ensuring our approach works in the real world.

**2.4. Connecting to "Transformer Components" & "Architecture":**

- The Encoder (Hiera):

The Encoder takes raw input (like text tokens) and creates "Hidden states" or context. AndSAM 2 uses a Hierarchical Vision Transformer (Hiera) as its Image Encoder. Instead of tokenizing text, it tokenizes the video frame pixels. Just like the Encoder in your slide's Figure 3-1, it processes the input once to create a rich feature representation (embedding) that the decoder can query later.

- The Decoder (Mask Decoder):

The Decoder taking "Token embeddings" and "Hidden states" from the encoder to predict the "Next word". And SAM 2's Mask Decoder (Figure 8) takes the image embeddings (from the encoder) and user prompts (clicks/boxes). Instead of predicting the "next word," it predicts the "mask."


**2.5. Connecting to "Formal Algorithms" (Attention Mechanisms):**


- Self-Attention ($X$ attending to $X$):

Self-attention is defined as relating tokens within the same sequence to one another9. In SAM 2 (Figure 8), the block labeled "self attn." in Figure 8 allows the prompt tokens (e.g., if you click five times on a car) to communicate with each other. This helps the model understand that Click 1 and Click 2 are related parts of the same object.

- Cross-Attention ($X$ attending to $Z$):

The cross-attention (often in Encoder-Decoder blocks) is where the query comes from one sequence and keys/values come from another.In SAM 2 (Figure 8), the blocks labeled "token to image attn" and "image to token attn" are Cross-Attention layers.

- Query ($Q$): The user prompts (tokens).

- Key/Value ($K, V$): The image embeddings from the Image Encoder.

- This is how the model "looks" at the specific part of the image you asked about.


**2.6. Connecting to "Positional Embeddings" (RoPE):**

The "Positional embeds" are added to token embeddings so the model knows the order of words13. Standard Transformers use absolute positions ($1, 2, 3...$). And SAM 2 uses Rotary Positional Embeddings (RoPE) (specifically 2D-RoPE) for its memory attention.

- Why it matters: Standard embeddings struggle when sequence lengths change (like in long videos). RoPE (a concept likely covered in your "Llama 3.2 Deep-Dive" unit) encodes relative positions using rotation matrices. This allows SAM 2 to understand that "Frame 50 is 10 frames after Frame 40," regardless of how long the video is, facilitating the "Streaming Memory" architecture.
  

# 3. Implementation & Demos

**3.1. The Setup:** We use the sam2 Python library with the Hiera-Large checkpoint for maximum accuracy.

**3.2. Code Walkthrough:**
```
import torch
from sam2.build_sam import build_sam2_video_predictor

# 1. Load the Model (Applying Pre-trained Weights)
checkpoint = "./checkpoints/sam2_hiera_large.pt"
predictor = build_sam2_video_predictor("sam2_hiera_l.yaml", checkpoint)

# 2. Initialize Video State
video_path = "./demo_video.mp4"
inference_state = predictor.init_state(video_path=video_path)

# 3. User Interaction (The Prompt)
# We provide a single positive click (Label 1) at coordinates X=200, Y=300 on Frame 0
_, _, mask_ids = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=[[200, 300]],
    labels=[1],
)

# 4. Propagation Loop
# The model iterates through the video, updating its Memory Bank at each step
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    # Visualize the mask overlay on the original frame
    show_mask(out_mask_logits, out_frame_idx)
```

Explanation:

- Model Initialization: We load the sam2_hiera_large checkpoint. This loads the Hiera (Hierarchical) image encoder weights, which allows the model to understand visual features at 44 FPS.

- State Management: The init_state function processes the video frames into embeddings. This creates the initial "Memory Bank" structure (FIFO queue) referenced in the methodology.

- Prompting: The add_new_points function simulates user interaction. Here, we provide a single positive click (labels=[1]) at specific coordinates on Frame 0. The model uses the Mask Decoder to generate an initial mask for the object at this frame.

- Temporal Propagation: The propagate_in_video loop is where the Memory Attention kicks in. It iterates through the remaining video frames. For every new frame, it queries the memory of the object from Frame 0 (and subsequent frames) to predict where the object has moved, handling occlusions and rotation automatically.


# 4. Assessment & Evaluation:


**4.1.Model Version & Architecture:** 
- Version: SAM 2 (and SAM 2.1).

- Architecture:  A unified Transformer architecture with a streaming memory mechanism for real-time video processing.

- Core Components:

Image Encoder:  Uses a Hiera (Hierarchical Vision Transformer) backbone to process frames.

Memory Attention: Allows the model to attend to past frame features and predictions stored in a Memory Bank (FIFO queue).

Mask Decoder:  Generates segmentation masks based on user prompts (clicks/boxes) and memory context.

We evaluate SAM 2 against previous State-of-the-Art (SOTA) models on three standard industry datasets to prove reliability.


**4.2. Intended Uses & License:**

- Primary Use: Promptable visual segmentation for images and videos (e.g., visual editing, robotic perception, data annotation).

- License: Released under the permissive Apache 2.0 License.

- Dataset License: The SA-V dataset is released under CC BY 4.0.

- Prohibited Uses:*Surveillance, military applications, or generating biometric/sensitive personal data without consent.


**4.3. Ethical & Bias Considerations:** 

- Fairness Evaluation: The authors conducted a fairness evaluation on the SA-V dataset and found minimal performance discrepancy across perceived gender and age groups.

- Geographic Diversity: The training data (SA-V) includes videos from 47 countries to ensure the model generalizes well across different global environments.

- Limitations: The model may still reflect biases present in the training data, and users are advised to perform their own fairness evaluations for specific use cases.



# 5. Model & Data Cards:
**Model Name:** Segment Anything Model 2 (SAM 2)

**Developer:** Meta FAIR (Fundamental AI Research)

**Release Date:** August 2024

**Training Data:** Trained on SA-V. It contains 51,000 videos and 643,000 masklets. It is 53x larger than previous datasets (like DAVIS), ensuring high reliability across different environments.


# 6. Critical Analysis:

**6.1.Revelations:** 


We analyzed the zero-shot performance of SAM 2 against strong baselines (SAM + XMem++ and SAM + Cutie). As shown in Figure 5 below, SAM 2 consistently achieves higher segmentation quality across all interaction steps.
![Zero-shot Accurancy](./images/fig5.png)

The Comparison:

- SAM 2 (Blue Line): Consistently achieves the highest accuracy across all interaction steps in both settings. It starts higher and improves steadily as more frames are annotated.

- SAM + Cutie (Green Dotted Line): A strong baseline but consistently underperforms SAM 2.

- SAM + XMem++ (Orange Dashed Line): The lowest performing of the three, though it still improves with more interactions.

- Key Takeaway: SAM 2 provides significantly better segmentation accuracy with fewer interactions compared to state-of-the-art baselines.
 

**6.2.Impact of the Results:** 
The quantitative results reveal a significant advancement in handling complex video scenarios:

| Dataset | Focus | SAM 2 Score (J&F) | vs. Previous Best |
| :--- | :--- | :--- | :--- |
| **DAVIS 2017** | Standard Benchmark | **90.7%** | +2.6% Improvement |
| **MOSE** | **Complex Occlusions** | **77.9%** | **+6.2% Improvement** |
| **YouTube-VOS** | Diverse Videos | **89.3%** | +1.8% Improvement |

This table suggests that the Memory Attention mechanism is the key differentiator. The 6.2% improvement on the MOSE dataset (which focuses on disappearing/reappearing objects) empirically proves that SAM 2's memory bank handles occlusions significantly better than previous state-of-the-art methods.

**6.3. Limitations & Future Directions:** 

- Addressing Current Limitations: As noted, the model currently struggles with "ID Switching" in crowded scenes (e.g., similar-looking people) and thin, fast-moving objects due to motion blur. A future step is to implement temporal smoothing post-processing to reduce jitter in these edge cases.

- Domain-Specific Fine-Tuning: While SAM 2 is a generalist model, it can be fine-tuned for specialized tasks. The next logical step is to train the model on medical datasets (e.g., tracking tumors in ultrasound video) or autonomous driving footage to improve reliability in safety-critical environments.

- Edge Deployment: Currently, the model requires significant GPU power (Hiera-Large). Future work involves quantizing the model (reducing precision from float32 to int8) to allow it to run in real-time on mobile devices and edge cameras.



# 7. Conclusion: 

This project demonstrates that SAM 2 successfully bridges the gap between static image segmentation and dynamic video processing. By treating video as a continuous stream rather than isolated 3D blocks, the Memory Attention mechanism solves the long-standing challenge of "Temporal Inconsistency," allowing the model to track objects even when they rotate, change lighting, or disappear behind occlusions. Our evaluation confirms that SAM 2 outperforms strong baselines like SAM+XMem++ and SAM+Cutie, achieving a 90.7% J&F score (Jaccard & F-measure) on DAVIS 2017 and a critical 6.2% improvement on the complex MOSE dataset. Ultimately, by unifying the architecture for both images and video, SAM 2 democratizes high-end visual effects and opens new possibilities for real-time robotic perception and automated video editing.


# 8. Documentation & Resource Links:

**8.1. Setup Instructions To run this project locally:**

To run this project locally, you need to install PyTorch and the official SAM 2 library.

**1. Install Dependencies. **
First, install PyTorch and TorchVision (the libraries for Computer Vision).
```bash
pip install torch torchvision
```
**2. Clone the Repository Pull the official code from Meta's GitHub.**

```bash
git clone [https://github.com/facebookresearch/segment-anything-2.git](https://github.com/facebookresearch/segment-anything-2.git)
cd segment-anything-2
```

**3.Install the SAM 2 library in editable mode so Python can find it.**

```bash
pip install -e .
```

**4.Download the pre-trained model weights ("Hiera Large") that we use for the demo.**
```bash
cd checkpoints
./download_ckpts.sh sam2_hiera_large
cd ..
```


**8.2. Resource Links:**

Primary Paper: Ravi, N., et al. (2024). SAM 2: Segment Anything in Images and Videos. Meta FAIR.\
https://arxiv.org/pdf/2408.00714

Dataset: The SA-V Dataset. Meta AI. \
https://ai.meta.com/datasets/segment-anything-video/

Codebase: Official GitHub Repository. \
https://github.com/facebookresearch/segment-anything-2




