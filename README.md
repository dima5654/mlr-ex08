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

Browser image decoding and resize interpolation can cause small numeric differences. The reference checker reports every actual/expected value and absolute error, with an absolute tolerance of 0.005. It does not replace model output with expected values.

## Verify

Click **Run reference check**, or open `?selftest=1`. This runs `check_image.jpg` through the same preprocessing and model as camera frames, and compares with `check_expected.json`. The expected probabilities are apple 0.70185935, cherry 0.28628042, chestnut 0.01177154, maple 0.00008872. A completed result is also exposed as `window.referenceResult`.

The reference inference and iPhone Safari behavior have not been verified in this restricted environment. A live deployment and independent QR decode are also pending. On the deployed page, `window.qrEncodedURL`, the footer link, and `#qr.dataset.encodedUrl` identify the QR payload. Scan the code on another device to independently verify it opens the same live page.

## Publish with gh

Run `./publish.ps1` in PowerShell from this folder after GitHub CLI networking/authentication are available. It creates the public `mlr-ex08` repository, writes the actual page URL into this README, pushes main, enables Pages from main at `/`, invites `MLRteacher` with push access, and waits for the page to respond successfully. An invitation still requires acceptance by MLRteacher.

Dependencies: [ONNX Runtime Web deployment](https://onnxruntime.ai/docs/tutorials/web/deploy.html), [QR generator](https://github.com/kazuhikoarase/qrcode-generator). QR pixels are generated locally in the browser; no external QR-image service is used.



