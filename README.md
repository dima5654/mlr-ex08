# Leaf Lens

A single HTML page using plain JavaScript and ONNX Runtime Web from a pinned CDN. No build step or image uploads to a server.

## Public page

Public page: [https://dima5654.github.io/mlr-ex08/](https://dima5654.github.io/mlr-ex08/). The page generates its QR code in the browser from its actual GitHub Pages URL, including the repository path. It shows the same URL as a clickable link.

## Use

For a desktop preview, run `powershell -ExecutionPolicy Bypass -File .\serve.ps1` in this folder and open http://localhost:8000/. Keep the terminal open. Double-clicking index.html uses `file://`, which browsers prevent from fetching the model and JSON files. The preview still needs internet access for CDN dependencies. iPhone camera use requires the deployed HTTPS page.

Open the HTTPS page in Safari on iPhone. Tap **Start rear camera**, allow camera access, and centre the leaf. Each inference uses the full central square of the frame. The inner guide helps positioning; it does not change the crop. **Stop camera** releases the camera. Choosing a photo also stops the camera. Returning from the background requires tapping Start again.

Photo upload is available when the camera is unavailable. All four probabilities update after each inference. ONNX Runtime uses single-threaded WebAssembly, avoiding a dependency on cross-origin isolation or WebGPU.

## Exact preprocessing

Settings and class order are loaded from `model_info.json`: apple, cherry, chestnut, maple; input size 96; RGB mean `[0.485, 0.456, 0.406]`; standard deviation `[0.229, 0.224, 0.225]`.

1. Crop the central square using the smaller source dimension.
2. Resize to 96 Ã— 96 using the browser canvas with high-quality smoothing.
3. Divide each RGB byte by 255, then compute `(value - mean[channel]) / std[channel]`.
4. Pack float32 values in NCHW order, shape `[1, 3, 96, 96]`.
5. Apply numerically stable softmax to the model's `linear_1` logits.

JPEG files (including the reference image and uploaded JPEGs) are decoded with the same pinned [MozJPEG WASM decoder](https://github.com/jamsinclair/jSquash/tree/main/packages/jpeg) on every browser, avoiding differences between Safari and desktop native JPEG decoders. A 96 x 96 JPEG feeds those decoded pixels directly into normalization without a canvas colour conversion. Other sizes still use the same central crop and canvas resize; other image formats use browser decoding. Live video consumes decoded camera frames directly. Resize interpolation can cause small numeric differences. The reference checker reports every actual/expected value and absolute error, with an unchanged absolute tolerance of 0.005. It also shows a fingerprint of the actual input tensor to help distinguish decoding differences from runtime differences. It does not replace model output with expected values.

## Verify

Click **Run reference check**, or open `?selftest=1`. This runs `check_image.jpg` through the same preprocessing and model as camera frames, and compares with `check_expected.json`. For the selected Task 4 model, the new expected probabilities are apple 0.09914806, cherry 0.85627663, chestnut 0.03354529, maple 0.01103004. A completed result is also exposed as `window.referenceResult`.

The original Task 2 model passed laptop and phone reference checks after the JPEG decoder correction, with the same input fingerprint `2c5a843a`. Manual upload matched too, and the owner verified the QR URL. That model's four original assets are preserved in [task2_original/](task2_original/); the earlier results and screenshot in [REVIEW.md](REVIEW.md) refer to that model.

The page now uses the Task 4 combined MobileNet, seed 0 (geometric and colour augmentation, cosine scheduling, 30 epochs). Across three seeds it averaged 71.43% normal validation and 73.81% camera-like validation accuracy. [REPORT.md](REPORT.md) contains the experimental measurements and remaining report work. New-model browser checks and real phone tests are pending. Model asset URLs include a version to avoid reusing the old cached model or expected values. The reference image is from the apple folder; this model predicts cherry for it, so a passing consistency check does not mean the prediction is correct.

## Publish with gh

Run `./publish.ps1` in PowerShell from this folder after GitHub CLI networking/authentication are available. It creates the public `mlr-ex08` repository, writes the actual page URL into this README, pushes main, enables Pages from main at `/`, invites `MLRteacher` with push access, and waits for the page to respond successfully. An invitation still requires acceptance by MLRteacher.

Dependencies: [ONNX Runtime Web deployment](https://onnxruntime.ai/docs/tutorials/web/deploy.html), [QR generator](https://github.com/kazuhikoarase/qrcode-generator). QR pixels are generated locally in the browser; no external QR-image service is used.



