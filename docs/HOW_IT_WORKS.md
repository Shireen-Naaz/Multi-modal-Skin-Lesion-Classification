# How this project works

A plain-language guide. If you can answer the questions at the end without looking, you understand the project.

## 1. The problem
Given a photo of a skin lesion, predict which of 7 diagnoses it is. Missing a melanoma is far worse than a false alarm, so the metric choice matters.

## 2. The data
HAM10000 has about 10,000 images plus a table with `age`, `sex`, `localization` (body site), `dx` (the diagnosis, our label) and `lesion_id`. Two-thirds of the images are `nv` (benign moles). A model that always says `nv` would score about 67 percent accuracy and be useless.

## 3. Why the split is by lesion
Several images can show the same lesion. If one photo of a lesion is in training and another is in testing, the model is partly being tested on something it has already seen, and the score is too high. Splitting on `lesion_id` keeps all photos of a lesion on one side.

## 4. Clinical features
- **Median imputation:** a few ages are missing; the training median fills them.
- **Standardizing age:** puts age on a similar scale to the other inputs so training is stable.
- **One-hot encoding:** turns categories such as `sex` and `localization` into 0/1 columns, because a network cannot use text.
- **Fit on train only:** the median, scaler and encoder learn from training data only. Learning them from test data would leak information.

## 5. The image branch
A CNN learns visual patterns (edges, colour, texture, shape). **EfficientNetB0 pretrained on ImageNet** already knows general visual features, so it needs far less data than training from scratch. This is **transfer learning**. **Augmentation** (random flips, rotations, zoom, contrast) shows the model varied versions of each image and reduces overfitting.

## 6. The clinical branch and fusion
The clinical vector goes through a small dense network. The image features (from global average pooling) and the clinical features are **concatenated** into one vector, then a dense layer and a softmax produce 7 probabilities. This is **feature-level (intermediate) fusion**.

## 7. Class imbalance
**Class weights** make mistakes on rare classes cost more in the loss, so the model cannot ignore them. Weights are computed from training frequencies.

## 8. Two-phase training
- **Phase 1:** the pretrained backbone is frozen; only the new layers learn (learning rate 1e-3).
- **Phase 2:** the top 40 backbone layers are unfrozen and fine-tuned at 1e-5. A tiny learning rate avoids destroying what the backbone already knows.
- **BatchNorm stays frozen** during fine-tuning, because small batches make its statistics unstable.
- **Early stopping** halts training when validation loss stops improving and restores the best weights.

## 9. Metrics
- **Accuracy** is misleading with imbalance.
- **Recall** per class: of the true cases of a class, how many the model found.
- **Macro recall / macro F1:** average over classes, so rare classes count as much as `nv`.
- **Melanoma recall:** the clinically most important number.

## 10. Image-only vs multimodal
Both models use the same split and settings. The difference in the results table tells you whether the clinical data helped. If it did not help much, that is still a valid, reportable finding.

## Self-check questions
1. Why split by `lesion_id` instead of by image?
2. Why fit the scaler and encoder on training data only?
3. Why is 67 percent accuracy not impressive here?
4. What does fusing features by concatenation mean?
5. Why freeze the backbone first, then fine-tune with a very small learning rate?
6. What do class weights do, and what do they not fix?
7. Why report melanoma recall separately?
8. Name two limitations of this project.

### Answers
1. Same-lesion photos in train and test leak information and inflate scores.
2. Using test data would leak information about it into preprocessing.
3. Always predicting `nv` already gets about 67 percent.
4. The image feature vector and the clinical feature vector are joined into one longer vector before the classifier.
5. The new layers start random; training them first stops large gradients from wrecking the pretrained weights, and the small rate then adapts the backbone gently.
6. They raise the loss for rare-class errors; they do not add information, so very rare classes stay hard.
7. A missed melanoma is the most costly error, and macro averages can hide it.
8. For example: one dataset with no external test, and few examples of rare classes.
