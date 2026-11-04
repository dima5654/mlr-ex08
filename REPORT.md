# Task 4 - A better leaf classifier

My app: https://dima5654.github.io/mlr-ex08/

## What I tried

I used the agent to add experiments to my existing notebook. The instruction was to keep our simple, readable code: start from Task 3 and change only what each experiment needs, with no rewrite, new framework or unnecessary classes. The experiments reuse my dataset, network definitions and training loop.

My first baseline was the Task 3d network, LeafNetBN, with geometric augmentation. It used Adam with a learning rate of 0.001, batches of 32 and 15 epochs. I first tested colour augmentation, label smoothing and a cosine learning-rate scheduler separately. Then I tested combinations and compared them with pretrained MobileNet from Task 2.

## How I compared them

All runs used the same 101 training images and 28 validation images. I kept the original split and repeated every variant with seeds 0, 1 and 2. I used the final epoch's weights for every run. The repeated baseline gave the same results in both rounds.

For the camera-like check, the validation images were randomly cropped, rotated and given colour changes once, using seed 2026. The saved set was reused for every run, including the second round. None of these images was used for training.

The colour change was ColorJitter with brightness, contrast and saturation 0.2, and hue 0.05. Label smoothing was 0.1. The cosine scheduler reduced the learning rate over the run. For the frozen MobileNet, only the new last layer was trained; the pretrained layers and their batch-normalization statistics stayed fixed.

## Results

The numbers are accuracy percentages, shown as mean +/- standard deviation across three seeds. The standard deviation shows how much the training runs varied; it is not a confidence interval.

| Model / change | Normal validation | Camera-like validation |
| --- | ---: | ---: |
| Task 3 baseline | 44.05 +/- 8.99 | 46.43 +/- 9.45 |
| Baseline + colour | 50.00 +/- 9.45 | 54.76 +/- 14.87 |
| Baseline + label smoothing | 54.76 +/- 8.25 | 52.38 +/- 2.06 |
| Baseline + cosine | 58.33 +/- 8.99 | 53.57 +/- 3.57 |
| Baseline + colour + cosine | 54.76 +/- 4.12 | 55.95 +/- 5.46 |
| Baseline + smoothing + cosine | 59.52 +/- 8.99 | 54.76 +/- 4.12 |
| Baseline + all three | 52.38 +/- 2.06 | 57.14 +/- 7.14 |
| MobileNet, full fine-tuning | 72.62 +/- 5.46 | 61.90 +/- 8.99 |
| MobileNet, frozen | 67.86 +/- 3.57 | 66.67 +/- 5.46 |
| MobileNet + geometry | 65.48 +/- 10.91 | 70.24 +/- 5.46 |
| MobileNet + colour | 63.10 +/- 2.06 | 61.90 +/- 2.06 |
| MobileNet + cosine | 64.29 +/- 9.45 | 57.14 +/- 12.88 |
| MobileNet, 30 epochs only | 53.57 +/- 10.71 | 55.95 +/- 7.43 |
| MobileNet + geometry + colour + cosine, 30 epochs | 71.43 +/- 3.57 | 73.81 +/- 7.43 |

The MobileNet single-change rows are compared with full fine-tuning for 15 epochs without augmentation. The last row combines several changes, so I cannot say which one caused its improvement. Changing from the small CNN to pretrained MobileNet is a separate comparison, not just one small change to the CNN.

The individual runs are saved in [round 1 results](task4_measurements.csv) and [round 2 results](task4_round2_measurements.csv). The settings are in [round 1 settings](task4_settings.json) and [round 2 settings](task4_round2_settings.json).

## What helped and what did not

For the small network, cosine gave the best normal-validation result among the single changes. Colour gave the best camera-like average, but it also varied the most: it actually made both scores worse for seed 0. Adding all three changes did not give the best score on both sets, so more changes were not always better.

MobileNet did better than the small network overall. Full fine-tuning was better on normal validation than freezing the network, but freezing was better on the camera-like set. Geometry improved MobileNet's camera-like average from 61.90% to 70.24%.

Some changes did not help MobileNet on their own. Colour did not change its camera-like average and made normal accuracy worse. Cosine alone and 30 epochs alone made both averages worse. Longer training was not automatically better on this small dataset.

The combined MobileNet had the highest camera-like average, 73.81%. This was 11.90 percentage points above plain MobileNet, while normal accuracy was 1.19 points lower.

## How much could be noise?

There are only 28 validation images, so getting one more image right changes accuracy by 3.57 percentage points. There are also different numbers of images per class: 4 apple, 10 cherry, 6 chestnut and 8 maple in validation.

The combined MobileNet's camera-like gain over plain MobileNet was 0, 7.14 and 28.57 points across the three seeds. That is promising, but not consistent enough to call it a certain improvement. Compared with geometry alone, the average gain was only 3.57 points. I would need more independent images to be confident about that difference. I used this set to choose the model, so it is no longer an independent final test.

## Model on my phone

With the agent, I chose the combined MobileNet for the phone because it had the best camera-like average. I used seed 0 rather than picking the highest-scoring seed. Its camera-like score was the middle of the three runs: 20/28, compared with 19/28 and 23/28 for the other seeds. Its normal score was 19/28.

I ran the export notebook in Colab. It checked the ONNX output against PyTorch and generated new expected probabilities. The original Task 2 model and its files are kept in task2_original/.

The new model passed the laptop reference check. All probabilities agreed to three decimals, with a largest reported error of 0.00000132. After the GitHub Pages deployment finished, I opened it on my phone, saw the new reference prediction of about 85% cherry, and checked that live camera detection worked. I did not record inference time or measure real-camera accuracy on a labelled set.

There is still a clear mistake: check_image.jpg is from the apple folder, but the new model predicts cherry. Passing the reference check only means the browser matches the exported model. It does not mean the prediction is correct. I would use this model as the next version to test on the phone, but I would collect more leaf photos before claiming it is reliably better.
