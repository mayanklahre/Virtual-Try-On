# Virtual Try-On

GPU-powered virtual try-on built with [CatVTON](https://github.com/Zheng-Chong/CatVTON) and deployed on [Modal](https://modal.com). Upload a person photo and a garment image to generate a try-on result.

## What You Need

- A Modal account and authenticated Modal CLI
- Python 3.10 or newer on your computer
- A Modal workspace with access to an A10G GPU

All ML and image-processing dependencies are installed inside the Modal image. Locally, you only need the Modal CLI.

## Set Up

Clone the repository and enter it:

```bash
git clone https://github.com/mayanklahre/Virtual-Try-On.git
cd Virtual-Try-On
```

Install and authenticate the Modal CLI:

```bash
python3 -m pip install modal
modal setup
```

Create the server-side login secret. These credentials protect both the web page and its `/try-on` endpoint; do not put them in source code or browser JavaScript.

```bash
modal secret create vto-web-auth \
  VTO_USERNAME='your-login-name' \
  VTO_PASSWORD='a-long-unique-password'
```

To replace existing credentials, add `--force` to that command.

## Download Model Weights

Run this once for each Modal environment. It downloads the CatVTON, Stable Diffusion, and human-parsing weights into the persistent `vto-model-weights` Modal Volume.

```bash
python3 -m modal run app.py::download_weights
```

This download is several gigabytes and can take a while. Future deploys reuse the Volume and do not need to download the weights again.

## Run It

For a hot-reloading development endpoint:

```bash
python3 -m modal serve app.py
```

For a deployed endpoint:

```bash
python3 -m modal deploy app.py
```

Modal prints the web URL after either command. Open it in a browser and enter the username and password saved in `vto-web-auth`.

You can also perform one command-line test using the sample images in `data/`:

```bash
python3 -m modal run app.py
```

The resulting image is written to `vto_result.png`.

## Image Guidance

The source images have a major effect on try-on quality.

- Person photo: use a front-facing image with the full upper body visible, arms separated from the torso, even lighting, and a simple background.
- Garment image: use one front-facing garment on a plain light background or a true transparent PNG/WebP. Keep the entire garment in frame.
- Use licensed, watermark-free product images. Watermarks and product-photo backgrounds can be learned by the model and appear in the result.
- The app accepts images up to 10 MB.

## API

The web UI calls `POST /try-on` with multipart form fields named `person` and `garment`. HTTP Basic authentication is required.

```bash
curl --user 'your-login-name:your-password' \
  -X POST 'https://YOUR_MODAL_URL/try-on' \
  -F 'person=@/path/to/person.jpg' \
  -F 'garment=@/path/to/garment.png' \
  --output vto_result.png
```

The public endpoint is limited to five try-on requests per client IP per hour to control GPU cost.

## External Frontends

The built-in UI is served by Modal. If a separate site needs to call the endpoint directly, update `ALLOWED_ORIGINS` in `app.py` with that site's exact HTTPS origin, then redeploy.

## Troubleshooting

- `Modal Secret 'vto-web-auth' not found`: create the secret in the same Modal environment used for deployment.
- `429 Rate limit exceeded`: wait for the current one-hour window to reset.
- Output includes background or edge artifacts: retry with a clean, front-facing garment image that has no watermark and a simple/transparent background.
- First request is slow: Modal may need to start a GPU container and load the model. Subsequent requests handled by the same container are faster.

## Deployment Notes

The application uses the `vto-model-weights` Modal Volume for weights and the `vto-rate-limit` Modal Dict for rate-limit slots. Keep these names unchanged unless you intentionally want separate environments or model storage.
