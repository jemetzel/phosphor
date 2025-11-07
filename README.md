# PHOSPHOR 🔮🎛️

_What happens when a machine dreams to music?_

**PHOSPHOR** is a portable, AI-assisted lightshow rig designed for live music environments, outdoor installations, and experimental performance art. Built on Raspberry Pi, OpenCV, and Art-Net protocols, it translates audio and video stimuli into vivid, reactive lighting using a modular system of addressable LEDs and DMX fixtures.

---

## ✨ Features

- 🎵 Real-time audio-reactive lighting
- 🎨 Spectral analysis mapped to hue, saturation, and brightness
- 📷 Visual input via Pi Camera for motion-sensitive effects
- 🕹️ Touchscreen interface for live control
- 🌐 Art-Net and WLED output support
- 🧠 ML-ready design (supports future model fine-tuning)

---

## 🧰 Tech Stack

- Python 3.11
- `numpy`, `scipy`, `librosa`, `opencv-python`, `sacn`, `python-osc`
- Raspberry Pi OS (Bookworm / Trixie)
- Custom DMX controller via USB-DMX or Art-Net
- WLED-compatible LED cubes, strips, or panels

---

## 🚀 Getting Started

git clone https://github.com/yourusername/phosphor.git

cd phosphor

pip install -r requirements.txt

python3 launch.py

_Make sure your DMX controller is connected and your fixture is addressed properly in config.json. WLED devices should be discoverable via mDNS or set via IP._

---

## 📦 Folder Structure

phosphor/

├── core/           # Signal processing + control logic

├── ui/             # Touchscreen interface & local preview

├── output/         # DMX, Art-Net, and WLED backends

├── assets/         # Sample loops, config files, docs

└── launch.py       # Main runtime entry point

---

## 🎭 Use Cases

- Intimate electronic music shows
- Urban projection art
- Ambient background loops for gallery spaces
- Reactive installations in natural settings
- VJ add-on for DJ performances

---

## 📸 Demo

🌐 Watch the 60-second video: https://koanzone.net/phosphor/demo

📷 Shot at Northerly Island, Chicago

💡 Featuring WLED HyperCube + PAR light + music-reactive controller

---

## 🧪 Roadmap

- MIDI sync support
- Model-based mood detection
- Remote preset uploads
- Multi-device mesh sync via OSC

---

## 🤝 Credits

Designed and engineered by Jeffrey Ege‑Koç Metzel

With help from Echo, the machine in the margins.

---

## 🛰 License

MIT License — fork, remix, glow freely.

---
