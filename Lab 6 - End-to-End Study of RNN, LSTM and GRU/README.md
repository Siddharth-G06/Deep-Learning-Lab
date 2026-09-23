# Experiment 6: End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

**Course:** CS3807 - Deep Learning Laboratory, Shiv Nadar University Chennai  
**Academic Year:** 2026-27  
**Semester:** V  

---

## Objective

The objective of this experiment is to develop an end-to-end understanding of recurrent sequence learning by implementing and comparing Vanilla RNN, LSTM and GRU models. The experiment also introduces Backpropagation Through Time (BPTT), the limitations of conventional RNNs, and the use of CNN-extracted features with recurrent networks for video understanding.

The experiment uses the UCI Human Activity Recognition Using Smartphones dataset as the primary sequence dataset. Students prepare temporal input sequences, visualise them, train RNN/LSTM/GRU models, compare their performance, analyse convergence and generalisation, and extend the pipeline to video understanding using CNN feature extraction followed by an LSTM or GRU. A sequence-to-sequence (encoder-decoder) task on a synthetic reversal dataset is also included.

---

## Learning Outcomes

After completing this experiment, students will be able to:
- explain the architecture and operation of Vanilla RNN, LSTM and GRU;
- implement Backpropagation Through Time (BPTT) and articulate the vanishing and exploding gradient problems;
- build and train recurrent models for multi-class time-series classification;
- compare RNN, LSTM and GRU using accuracy, F1-score, confusion matrices and parameter count;
- analyse the effect of sequence length on model performance and computational cost;
- extract spatial features from video frames using a pretrained CNN and feed them to a recurrent model;
- implement and evaluate an encoder-decoder LSTM for sequence-to-sequence learning;
- distinguish between token accuracy and sequence accuracy; and
- draw meaningful inferences from training and validation curves, confusion matrices and comparison plots.

---

## Dataset Information

### Primary Dataset - UCI Human Activity Recognition Using Smartphones

| Property | Details |
|---|---|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| **Subjects** | 30 volunteers aged 19-48 |
| **Sensor** | Samsung Galaxy S II (accelerometer + gyroscope) |
| **Sampling Rate** | 50 Hz, sliding window of 128 readings (2.56 s), 50% overlap |
| **Input Features** | 561 time-domain and frequency-domain features |
| **Classes** | 6 (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) |
| **Training Samples** | 7,352 |
| **Test Samples** | 2,947 |

The raw sensor signals are reshaped into temporal sequences. Each sample contains readings from six sensor channels - three axes of acceleration and three axes of angular velocity - arranged over time.

### Secondary Dataset - Video Understanding (UCF101 subset)

A subset of the [UCF101 action recognition dataset](https://www.crcv.ucf.edu/data/UCF101.php) is used for the video understanding component. Short video clips are decoded into frames; a pretrained CNN (e.g., MobileNetV2 or VGG16) extracts spatial features per frame; and the resulting feature sequence is passed to an LSTM or GRU for temporal classification.

### Synthetic Dataset - Sequence-to-Sequence Reversal

A synthetic dataset of several thousand short integer sequences is generated in-code. The model learns to reverse input sequences (e.g., [1, 4, 7, 2] to [2, 7, 4, 1]) using an encoder-decoder LSTM architecture.

---

## Experimental Procedure

The implementation is documented in `Experiment6_Complete_Lab.ipynb` and is divided into the following key tasks:

1. **Dataset Loading and Sequence Preparation:** Loading the UCI HAR dataset, reshaping raw signals into temporal windows, visualising sensor signals over time (Plot 1), and printing dataset statistics.
2. **Recurrent Model Construction:** Building three models - Vanilla RNN, LSTM and GRU - using TensorFlow/Keras with identical architecture settings (32 recurrent units, same optimiser, loss and batch size) for a fair comparison.
3. **Model Training and Loss Visualisation:** Training all three models and plotting per-epoch training and validation loss curves (Plots 2a-2c) to study convergence and generalisation.
4. **Accuracy Visualisation:** Plotting training and validation accuracy curves for each model (Plots 3a-3c) to identify overfitting or underfitting.
5. **Confusion Matrix Analysis:** Evaluating each model on the test set and generating per-class confusion matrices (Plots 4a-4c) to identify activity-wise errors.
6. **Model Comparison:** Comparing RNN, LSTM and GRU on accuracy, macro F1-score, precision, recall and parameter count in a consolidated bar chart (Plot 5).
7. **Sequence Length Study:** Varying the sequence length and plotting macro F1-score against sequence length (Plot 6) to study the trade-off between context and computational cost.
8. **Video Understanding Pipeline:**
   - Decoding video clips into frames and visualising sample frames (Plot 7).
   - Extracting per-frame CNN features using a pretrained backbone.
   - Training LSTM and GRU on the feature sequences and plotting training/validation curves (Plot 8).
   - Generating a confusion matrix for video action classes (Plot 9).
9. **Sequence-to-Sequence Learning:** Building an encoder-decoder LSTM to learn the sequence reversal task. Evaluating using token accuracy, sequence accuracy, training loss and validation loss (Plot 10).

---

## Repository Contents

```
Lab 6 - End-to-End Study of RNN, LSTM and GRU/
+-- README.md                                        # This document
+-- Experiment6_Complete_Lab.ipynb                   # Jupyter Notebook with the full implementation
+-- Experiment_6(1).tex                              # Lab manual (LaTeX source)
+-- Experiment_6.pdf                                 # Lab manual (PDF)
+-- Sensor signal versus time.png                    # Raw sensor signal visualisation
+-- plot1_sensor_signals.png/.eps                    # Plot 1 - Sensor signals over time
+-- plot2_rnn_loss.png/.eps                          # Plot 2a - RNN training & validation loss
+-- plot2_lstm_loss.png/.eps                         # Plot 2b - LSTM training & validation loss
+-- plot2_gru_loss.png/.eps                          # Plot 2c - GRU training & validation loss
+-- plot3_rnn_accuracy.png/.eps                      # Plot 3a - RNN accuracy curves
+-- plot3_lstm_accuracy.png/.eps                     # Plot 3b - LSTM accuracy curves
+-- plot3_gru_accuracy.png/.eps                      # Plot 3c - GRU accuracy curves
+-- plot4_rnn_confusion_matrix.png/.eps              # Plot 4a - RNN confusion matrix
+-- plot4_lstm_confusion_matrix.png/.eps             # Plot 4b - LSTM confusion matrix
+-- plot4_gru_confusion_matrix.png/.eps              # Plot 4c - GRU confusion matrix
+-- plot5_model_comparison.png/.eps                  # Plot 5 - Model comparison bar chart
+-- plot6_sequence_length_vs_f1.png/.eps             # Plot 6 - Sequence length vs. F1-score
+-- plot7_video_sample_frames.png/.eps               # Plot 7 - Sample video frames
+-- plot8_video_lstm_curves.png/.eps                 # Plot 8a - Video LSTM training curves
+-- plot8_video_gru_curves.png/.eps                  # Plot 8b - Video GRU training curves
+-- plot9_video_confusion_matrix.png/.eps            # Plot 9 - Video confusion matrix
+-- plot_seq2seq_loss_and_position_accuracy.png/.eps # Plot 10 - Seq2Seq loss & accuracy
```

---

## Dependencies

The following Python libraries are required to run the notebook:

- `tensorflow` (>= 2.x)
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `opencv-python` *(for video frame extraction)*
- `jupyter`

Install all dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python jupyter
```

---

## Execution Instructions

1. Navigate to the experiment folder:
   ```bash
   cd "Lab 6 - End-to-End Study of RNN, LSTM and GRU"
   ```
2. Install the required libraries:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python jupyter
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Experiment6_Complete_Lab.ipynb
   ```
4. Execute the cells sequentially from top to bottom. Generated plots will be saved automatically in the same directory.

> **Note:** The video understanding component (Task 8) requires video data and may take additional time on CPU-only machines. Use a GPU-enabled environment or reduce the number of video clips for faster execution.

---

## Key Observations

- **Vanishing Gradient:** The Vanilla RNN struggles to capture long-range dependencies in the sensor signal. Training and validation accuracy plateau earlier than LSTM and GRU, and the loss curves show less smooth convergence.
- **LSTM vs GRU:** LSTM achieves slightly higher accuracy on the HAR task due to its separate cell state and forget gate, but GRU converges faster and uses fewer parameters owing to its simpler gating mechanism.
- **Sequence Length Effect:** Longer sequences provide more temporal context and improve macro F1-score up to a point; beyond that, computational cost increases while performance gains diminish.
- **CNN as Feature Extractor:** The pretrained CNN effectively captures spatial appearance features from each frame, removing the need for the recurrent layer to learn spatial patterns, allowing it to focus entirely on temporal dynamics.
- **Seq2Seq Gap:** Token accuracy is consistently higher than sequence accuracy because a single incorrectly predicted token renders the entire output sequence incorrect, even if all other tokens are right.
- **Confusion Patterns:** The confusion matrices reveal that activities involving similar body postures (e.g., SITTING and STANDING) are more likely to be confused by all three recurrent models.

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
3. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.
4. Anguita, D., et al. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. *ESANN*.
5. Soomro, K., Zamir, A. R., & Shah, M. (2012). UCF101: A Dataset of 101 Human Actions Classes From Videos in the Wild.
6. TensorFlow Documentation: https://www.tensorflow.org
7. Keras Documentation: https://keras.io