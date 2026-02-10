# Human Motion Detection

Lightweight human motion detection and tracking project using YOLOv8-pose, OpenCV and a simple tracker.

**Contents**
- `main.py` — entrypoint that runs the detection loop (webcam / video file).
- `app.py` — smoke-test runner to validate environment.
- `utils/` — detector and tracker helpers.
- `requirements.txt` — Python dependencies.

**Quick Start (macOS / Linux)**

1. Install Python 3.11 (Homebrew macOS example):

```bash
brew install python@3.11
```

CI/CD Workflows included
-----------------------

This repository includes two GitHub Actions workflows to help automate deployment:

- `.github/workflows/frontend-deploy.yml` — deploys the `frontend/` folder to Vercel on pushes to `main`. It uses the `amondnet/vercel-action` and requires these repository secrets:
	- `VERCEL_TOKEN` (required)
	- `VERCEL_ORG_ID` (optional)
	- `VERCEL_PROJECT_ID` (optional)

- `.github/workflows/backend-deploy.yml` — SSHs into a remote VM and runs `deploy_vm.sh` to update and start the backend. It requires these repository secrets:
	- `DEPLOY_HOST` — target VM IP/hostname
	- `DEPLOY_USER` — SSH username
	- `DEPLOY_SSH_KEY` — private SSH key (PEM)
	- `DEPLOY_SSH_PORT` — optional (default 22)

Using the workflows
-------------------

1. Add the required secrets to your GitHub repository: Settings → Secrets → Actions.
2. Connect the repository to Vercel if you prefer automatic frontend deploys via Vercel UI, or provide the `VERCEL_*` secrets to allow the workflow to trigger a Vercel deploy.
3. For backend deploys, ensure the target VM is reachable from GitHub Actions and that the provided `DEPLOY_SSH_KEY` has access to the `DEPLOY_USER` account. The workflow will clone or reset the `Human_motion_detection` directory and run `deploy_vm.sh`.

Notes and recommendations
-------------------------
- For large model files, prefer hosting them in S3 or another object store and download at runtime. Do not commit large binaries to Git history.
- If you need a secure CI/CD pipeline for production, consider restricting the SSH user, using a deployment user account, and adding firewall rules to limit access.

2. Create and activate a virtual environment (from the project root):

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

3. Upgrade build tools and install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. Run a quick smoke test to verify imports:

```bash
.venv311/bin/python app.py
```

5. Run the main detector (uses webcam `0` by default):

```bash
.venv311/bin/python main.py
```

Notes:
- The first run may download pretrained weights (YOLO assets). Keep an eye on the console.
- If you plan to process video files instead of a webcam, pass the path to `system.run(source='path/to/video.mp4')` or modify `main.py` accordingly.

Hosting and Deployment
----------------------

Below are hosting options you can choose from; pick one and I'll implement it:

1) Local service (recommended for a single machine)
- Run `main.py` inside the virtual environment and configure a background service:
	- macOS: create a `launchd` plist to run the project at login/boot.
	- Linux: create a `systemd` service that activates the `.venv311` and runs `main.py`.

2) Cloud VM (AWS / GCP / Azure)
- Provision a VM (CPU or GPU). Steps I can automate for you:
	- Create a startup script that installs Python 3.11, creates the venv, installs requirements, and runs the app as a system service.
	- If you need higher throughput, choose an instance with an NVIDIA GPU and I can provide the CUDA-compatible PyTorch install command.

3) Add a lightweight web UI (Flask or Streamlit)
- I can add a small server that runs the detection loop and streams annotated frames over HTTP or a WebSocket. Useful for remote monitoring and easy hosting on a VM or PaaS.

Notes & Troubleshooting
- Keep `pip`, `setuptools`, and `wheel` up-to-date before installing requirements.
- On macOS ensure camera permissions are enabled for the terminal/host app.
- For headless or remote runs, modify `main.py` to save annotated frames or emit them over the network instead of opening a GUI window.

Next steps I can take for you now
- Create and enable a `systemd` service (if you're on Linux) or a `launchd` plist (if on macOS).
- Add a small Flask/Streamlit web UI and route to stream frames.
- Prepare a cloud VM startup script and instructions (or deploy to a VM if you provide access).

Tell me which option you want me to implement and I will proceed.

Hybrid deployment (Vercel frontend + VM backend)
-----------------------------------------------

This project now includes a small hybrid scaffold:

- `frontend/index.html`: static frontend suitable for deploying to Vercel.
- `backend/api_server.py`: FastAPI backend that runs the detector in the background and serves the latest annotated frame at `/frame` and stats at `/status`.

How it works
- Deploy `frontend/` to Vercel (static site). The frontend polls `http://<BACKEND_HOST>:8000/frame` to display the latest frame.
- Run the `backend` on a cloud VM (or on your Mac) and make it reachable from the frontend (open port 8000 or use a reverse proxy / tunnel).

Quick deploy steps
1. Frontend (Vercel)
	- Create a new Vercel project and point it at this repo's `frontend` folder (Vercel autodetects static site). Deploy — Vercel will provide a URL.
	- In production, edit `frontend/index.html` to replace the `backendOrigin` assignment with your backend URL (e.g. `https://api.example.com`).

2. Backend (VM)
	- Provision an Ubuntu VM (or similar). On the VM:

```bash
# update + Python 3.11 install (example for Ubuntu)
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential

# clone repo and create venv
git clone <your-repo> ~/human_motion_detection
cd ~/human_motion_detection
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt

# run the backend (for testing)
uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

	- To run as a persistent service, create a `systemd` unit (example):

```ini
[Unit]
Description=Human Motion Detection Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/human_motion_detection
ExecStart=/home/ubuntu/human_motion_detection/.venv311/bin/uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Security / CORS:
- Open port 8000 or run behind a reverse proxy (nginx) and use HTTPS. Configure firewall rules to restrict access if needed.

If you want, I can:
- scaffold a proper Next.js frontend instead of static HTML for Vercel, or
- provision a cloud VM with a one-shot script (I can provide commands for AWS/GCP/DigitalOcean), or
- implement an authenticated streaming endpoint instead of simple polling.


macOS `launchd` service (one-click)
----------------------------------

I added a sample `launchd` plist (`com.human_motion_detection.plist`) and a small wrapper script `run_main.sh` to the project. To install and run the service on your macOS machine:

1. Make the wrapper executable (from project root):

```bash
chmod +x run_main.sh
```

2. Copy the plist to your LaunchAgents folder and load it:

```bash
cp com.human_motion_detection.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.human_motion_detection.plist
```

3. Check status and logs:

```bash
launchctl list | grep com.human_motion_detection
tail -n 200 hmd.log
tail -n 200 hmd.err
```

4. To stop or remove the service:

```bash
launchctl unload ~/Library/LaunchAgents/com.human_motion_detection.plist
rm ~/Library/LaunchAgents/com.human_motion_detection.plist
```

Notes:
- The plist runs the command in the project directory using the `.venv311` Python interpreter and writes logs to `hmd.log`/`hmd.err` in the project root. If your project path differs, update the plist's `ProgramArguments` line accordingly before copying it to `~/Library/LaunchAgents`.
- If you prefer the plist to call the `run_main.sh` wrapper instead, change the `ProgramArguments` to call `/bin/bash -lc "/Users/admin/Downloads/human_motion_detection/run_main.sh"` or adjust the plist file accordingly.

