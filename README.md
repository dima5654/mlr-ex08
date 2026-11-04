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

Click **Run reference check**, or open `?selftest=1`. This runs `check_image.jpg` through the same preprocessing and model as camera frames, and compares with `check_expected.json`. The expected probabilities are apple 0.70185935, cherry 0.28628042, chestnut 0.01177154, maple 0.00008872. A completed result is also exposed as `window.referenceResult`.

The owner reported passing reference checks on laptop and phone after the JPEG decoder correction: both returned the same probabilities and input fingerprint `2c5a843a`, with every probability agreeing with check_expected.json to three decimals. Manually uploading check_image.jpg was also confirmed to give the same results. The owner confirmed that scanning the QR opens the exact public Pages URL above. [REVIEW.md](REVIEW.md) includes the supplied phone screenshot of live camera classification, numerical results, preprocessing and privacy checks, and the remaining notebook comparison. Task 4 is deferred; the local REPORT.md draft is not a completed Task 4 submission.

## Publish with gh

Run `./publish.ps1` in PowerShell from this folder after GitHub CLI networking/authentication are available. It creates the public `mlr-ex08` repository, writes the actual page URL into this README, pushes main, enables Pages from main at `/`, invites `MLRteacher` with push access, and waits for the page to respond successfully. An invitation still requires acceptance by MLRteacher.

Dependencies: [ONNX Runtime Web deployment](https://onnxruntime.ai/docs/tutorials/web/deploy.html), [QR generator](https://github.com/kazuhikoarase/qrcode-generator). QR pixels are generated locally in the browser; no external QR-image service is used.



