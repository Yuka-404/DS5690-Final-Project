# Reality 2.0: Real-Time Video Object Editing (SAM 2)

Presenter: Kunyang Ji

Date: 2025.12.4

# 1. Overview:

The Problem: Computer vision has mastered static images, but video remains a challenge due to "Temporal Inconsistency." When editing video, if you identify an object (like a car) in Frame 1, current models often "lose" it by Frame 50 if it rotates, changes lighting, or goes behind a tree (occlusion). This makes automated video editing (VFX) impossible without manual frame-by-frame work.

Proposed Approach: This project implements Meta’s Segment Anything Model 2 (SAM 2). Unlike previous models that treat video as 3D blocks, SAM 2 treats video as a stream of images with a "Memory Bank." It allows users to click an object once, and the model propagates that selection through the entire video in real-time.


# 2. Methdology:

**2.1. 
Theory**: The "Memory Attention" Mechanism We apply the concept of Attention Mechanisms (from Transformer theory). In standard Transformers, attention looks at all tokens in a sentence. In SAM 2, we implement "Memory Attention," where the model attends to features from past frames stored in a FIFO (First-In-First-Out) queue.

Application: When the object disappears (occlusion), the model queries the Memory Bank for the object's last known embedding to re-identify it when it reappears.

**2.2. Architecture Pipeline We utilize a unified pipeline consisting of three components:**

Image Encoder (Hiera): We use a Hierarchical Vision Transformer to encode video frames into feature embeddings. This is an application of ViT (Vision Transformer) architectures but optimized for speed.

Memory Encoder: This module compresses current masks into "memory tokens" that are stored for future reference.

Mask Decoder: This component fuses the Current Frame Embedding + Memory Context + User Prompts (clicks) to predict the segmentation mask.

**2.3. Zero-Shot Generalization:** Leveraging "Foundation Model" theory, this project demonstrates Zero-Shot capabilities. We do not fine-tune the model on specific objects (like cars or people). Instead, we rely on its pre-trained "Promptable" nature to segment any object class based solely on geometric and texture cues.



# 3. Implementation &2 Demos

**3.1. The Setup:** We use the sam2 Python library with the Hiera-Large checkpoint for maximum accuracy.

**3.2. Code Walkthrough**
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

# 4. Assessment & Evaluation:
**4.1. Model Architecture**

Version: SAM 2 (Hiera-Large Backbone).

Parameter Count: ~224M parameters.

Inference Speed: ~44 FPS (Frames Per Second) on A100 GPU, validating "Real-Time" claims.

**4.2. Intended Uses & Licensing**

License: Apache 2.0 (Permissive Open Source).

Intended Use: Video editing, autonomous robotics tracking, and scientific data annotation (e.g., tracking cells in microscopy).

**4.3. Ethical & Bias Considerations**

Data Bias: The model was trained on the SA-V dataset, which was explicitly curated to include geographically diverse videos (from 50+ countries) to prevent Western-centric bias in object recognition.

Surveillance Risk: This technology enables highly accurate tracking of individuals in CCTV footage. We must acknowledge the dual-use nature of this tool for privacy invasion.



# 5. Model & Data Cards:
Model Name: Segment Anything Model 2 (SAM 2)

Developer: Meta FAIR (Fundamental AI Research)

Release Date: August 2024

Training Data: Trained on SA-V, the world's largest video segmentation dataset containing 51,000 videos and 643,000 masklets.


# 6. Critical Analysis:

**6.1. Impact:** The End of Rotoscoping This project reveals that manual video segmentation (rotoscoping), which historically took Hollywood studios weeks, is now a solved problem solvable in seconds. This lowers the barrier to entry for high-end VFX.

**6.2. Technical Insight:** Memory is Key The success of SAM 2 suggests that for Video AI, Architecture (Memory mechanisms) is more important than just Scale (more data). The ability to "remember" past tokens is what solves the occlusion problem, not just training on more videos.

**6.3. Limitations:** The model still struggles with "ID Switching" in crowded scenes (e.g., confusing two similar-looking people wearing the same shirt) and extremely thin, fast-moving objects (like spinning wheels) due to motion blur.

# 7. Conclusion: 

【need to edit】


# 8. Documentation & Resource Links:

**8.1. Setup Instructions To run this project locally:**

Install dependencies: pip install torch torchvision

Clone repo: git clone https://github.com/facebookresearch/segment-anything-2

Download checkpoint: cd checkpoints && ./download_ckpts.sh


**7.2. Resource Links：**

Primary Paper: Ravi, N., et al. (2024). SAM 2: Segment Anything in Images and Videos. Meta FAIR.

Dataset: The SA-V Dataset. Meta AI. https://ai.meta.com/datasets/segment-anything-video/

Codebase: Official GitHub Repository. https://github.com/facebookresearch/segment-anything-2

