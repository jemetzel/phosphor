import os, sys, time, math, socket, collections
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
os.environ.setdefault("OMP_NUM_THREADS" , "1")

import numpy as np
import pygame, cv2, sounddevice as sd

# CONFIG

SAFE_MODE = False
OUTPUTS_ENABLED = True
CAMERA_ENABLED = False
BLACKOUT = False

# speed knobs

LITE_MODE = True
SHOW_CAMERA_PREVIEW = True 
SHOW_CHROMA = False

SR = 22050
NFFT = 1024
HOP = 512
ANALYZE_SECONDS = 1/3
ANALYZE_EVERY_SEC = 1/30
BEAT_FALLBACK_SEC = 0.55

SACN_FPS = 20

# HyperCube

ENABLE_CUBE = True
CUBE_IP = "192.168.1.71"
CUBE_UNIVERSES = [1, 2]
PIXELS_PER_UNI = 144
CHANNELS_PER_PIXEL = 3

# Art-Net PAR
ARTNET_IP = os.getenv("ARTNET_IP", "10.201.6.101")
ARTNET_PORT = 6454
ARTNET_UNIVERSE = int(os.getenv("ARTNET_UNIVERSE", "0"))
PAR_ADDR = 1

# Camera via Streamer (Bookworm/libcamera)

if CAMERA_ENABLED:
GST_PIPE = (
'libcamerasrc ! video/x-raw,width=480,height=360,framerate=30/1,format=RGB '
'! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1'
)

W,H = 480,800

# UTILS

class EMA:
  def __init__(self, a=0.2):
    self.a=a; self.y=None
  def __call__(self, x):
    x=float(x)
    self.y = x if self.y is None else (1-self.a)*self.y + self.a*x
    return self. y

def hsv_to_rgb(h, s, v) :
  '''h in deg [0..360), s,v in [0.1] → (0..255 ints) '''
  h = (h % 360)/60.0
  c = v*s
  x = c* (1-abs((h%2)-1))
  m = v-c
  r,g,b = [(c,x,0), (x,c,0), (0,c,x), (0,x,c), (x,0,c), (c,0,x)][int(h)%6]
  return tuple(int(255*(m+u)) for u in (r,g,b))

def circ_mid (h1, h2):
  '''Circular midpoint between two hues (deg).'''
  d = ((h2 - h1 + 180) % 360) - 180
  return (h1 + 0.5*d) % 360

def circ_lerp(a, b, t):
  ''''Shortest-path hue mix (degrees).'''
  d = ((b - a + 180) % 360) - 180
  return (a + t*d) % 360

def clamp01 (x): return 0.0 if x<0 else 1.0 if x>1 else x

# Pitch-class → hue (deg), bins 0.. 11 as Librosa chroma (C..B)
# C: Green, C#: Green-cyan, D: Cyan, D#: Blue-cyan, E: Blue, F: Blue-violet, F#: Violet, G: Magenta,
# G#: Red, A: Orange, A#: Yellow, B: Yellow-Green
PCZHUE = np.array([120, 150, 180, 200, 220, 250, 280, 300, 0, 30, 60, 90], dtype=float)

#== ART-NET / SACN
class ArtNetSender:
    def __init__(self, ip, universe=0):
        self.addr = (ip, ARTNET_PORT)
        self.seq = 1
        self.universe = universe & 0x7FFF
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def send(self, dmx):
        if not OUTPUTS_ENABLED: return
        if BLACKOUT: dmx = bytes(512)
        self.seq = (self.seq + 1) & 0xFF or 1
        length = max(2, min(512, len(dmx)))
        hdr  = bytearray(b'Art-Net\x00')
        hdr += (0x5000).to_bytes(2, 'little')  # OpDmx
        hdr += (14).to_bytes(2, 'big')
        hdr += bytes([self.seq, 0x00])
        hdr += (self.universe & 0xFF).to_bytes(1, 'little')
        hdr += ((self.universe>>8)&0x7F).to_bytes(1, 'little')
        hdr += (length).to_bytes(2, 'big')
        self.sock.sendto(hdr + dmx[:length], self.addr)

ART = ArtNetSender(ARTNET_IP, ARTNET_UNIVERSE)

def par_rgb(r,g,b, strobe=0):
    dmx = bytearray(512)
    i = PAR_ADDR-1
    r = int(clamp01(r/255)*255); g = int(clamp01(g/255)*255); b = int(clamp01(b/255)*255)
    dim = max(r,g,b)
    dmx[i+0]=dim; dmx[i+1]=r; dmx[i+2]=g; dmx[i+3]=b
    dmx[i+4]=max(0,min(255,int(strobe))); dmx[i+5]=0; dmx[i+6]=0
    ART.send(dmx)

# sACN

cube = {}
for uni in CUBE_UNIVERSES:
cube[uni] = [0]*512

try:
    import sacn
    _cube_sender = sacn.sACNsender(); _cube_sender.start()
    for u in CUBE_UNIVERSES:
        _cube_sender.activate_output(u); _cube_sender[u].destination=CUBE_IP
        _cube_sender[u].multicast=False; _cube_sender[u].fps=SACN_FPS
    CUBE_OK = True
except Exception as e:
    print("Cube disabled:", e); CUBE_OK=False; _cube_sender=None
   
def cube_rgb(r,g, b,L) :
  for uni in CUBE_UNIVERSES:
  if uni == 1:
    for p in range(0, min(int(288*L),144)) :
      cube[uni][3*p:3*p+3] = [r,g,b]
    for p in range (min(int(288*L), 144),144):
      cube[uni][3*p:3*p+3] = [0,0,0]
  else:
    for p in range(0, max(0,int(288*L-144))):
      cube[uni][3*p:3*p+3] = [r,g,b]
    for p in range(max(0,int(288*L-144)),144):
      cube[uni][3*p: 3*p+3] = [0,0,0]
      _cube_sender[uni].dmx_data = bytes(cube[uni])

# CAMERA

_cap=None; _prev_gray=None
def cam_open():
    global _cap
    _cap = cv2.VideoCapture(GST_PIPE, cv2.CAP_GSTREAMER)
    if not _cap or not _cap.isOpened():
        print("Camera off, running without preview."); _cap=None

def vision_feats():
    """Returns (frame_small, room_h_deg, room_v_norm, motion_0_1)"""
    global _cap, _prev_gray
    if _cap is None: return None, 0.0, 0.0, 0.0
    ok, frame = _cap.read()
    if not ok or frame is None: return None, 0.0, 0.0, 0.0
    small = cv2.resize(frame, (160,120))
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = float(np.mean(hsv[:,:,0])) * 2.0        # 0..360
    v = float(np.mean(hsv[:,:,2])) / 255.0      # 0..1
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    H,W = gray.shape
    roi = gray[int(H*0.2):int(H*0.8), int(W*0.2):int(W*0.8)]
    if _prev_gray is None: motion=0.0
    else:
        diff = cv2.absdiff(roi, _prev_gray)
        diff = cv2.GaussianBlur(diff,(5,5),0)
        _,mask = cv2.threshold(diff,12,255,cv2.THRESH_BINARY)
        mask = cv2.dilate(mask,None,1)
        motion = float(np.mean(mask>0))
    _prev_gray = roi
    return small, h, v, motion

# -- Rolling stats helper from collections

class RollingStats:
  def __init__(self, seconds=5.0, fps=30):
    self.buf = deque(maxlen=max(2, int(seconds*fps)))
    self.warm = 0
  def push(self, x: float):
    self.buf.append(float(x)); self.warm += 1
  def mean_std(self) :
    if not self.buf: return 0.0, 1.0
    a = np.fromiter(self.buf, dtype=np.float32)
    m = float(a.mean()); s = float(a.std(ddof=0))
    return m, (s if s >= 1e-6 else 1e-6)
  def ready(self): 
    return self.warm >= max(15, self.buf.maxlen//4)

class IntensityController:
    """Tempo-aware intensity with hysteresis, holds, and smooth decay."""
    def __init__(self, bpm=120.0):
        self.bpm = max(60.0, min(180.0, float(bpm)))
        self.level = 0               # 0..3
        self.I = 0.0                 # continuous 0..1
        self.hold_until = 0.0
        # z-score thresholds (up/down) and holds in beats for each level
        self.levels = [
            dict(up=0.8,  down=0.5, hold_beats=0),  # to L1
            dict(up=2,  down=1,  hold_beats=1),  # to L2
            dict(up=2.8,  down=1.5,  hold_beats=4),  # to L3
        ]
        self.tau_beats = 0.8         # decay time constant (in beats)
        self.k_boost   = 0.10        # how much a strong onset bumps I

    def update(self, onset, Ostats, now):
        # Rolling means/SDs -> z-scores
        Ostats.push(onset)
        Om, Os = Ostats.mean_std()
        z = (onset - Om) / Os

        beat_sec = 60.0 / self.bpm
        # Decay continuous intensity between frames
        dt = 1/30.0
        self.I *= math.exp(-dt / (self.tau_beats * beat_sec))

        # Boost on strong onsets (makes I feel snappy)
        if onset > 0.003 and z > 0.6:
            self.I = min(1.0, self.I + self.k_boost * (z - 0.3))

        # Discrete level state with hysteresis & holds
        if now >= self.hold_until:
            if self.level < 3 and onset > 0.003 and z >= self.levels[self.level]['up']:
                self.level += 1
                self.hold_until = now + self.levels[self.level-1]['hold_beats'] * beat_sec
            elif self.level > 0 and z <= self.levels[self.level-1]['down']:
                self.level -= 1
                # no hold on downshift

        # Map level to a floor for I so it doesn't drop immediately after a trigger
        level_floor = [0.00, 0.33, 0.66, 0.9][self.level]
        self.I = max(self.I, level_floor)
       
        self.I = (self.I - (self.I % (1/24)))

        return self.I, self.level, z


# AUDIO / LIBROSA ANALYZER

# ring buffer for audio callback

from collections import deque

_audio_q = collections.deque(maxlen=2)
def _audio_cb(indata, frames, time_info, status):
  try: _audio_q.append(indata[:,0].astype(np.float32).copy())
  except: pass

def get_audio_chunk():
  if not _audio_q: return None
  return _audio_q.popleft()

# librosa analyzer
 
import librosa

class LibrosaAnalyzerLite:
  '''Single-STFT pipeline; chroma + percussive onset from one pass.'''
  def __init__(self, sr=SR, n_fft=NFFT, hop=HOP, seconds=ANALYZE_SECONDS):
    self.sr, self.n_fft, self.hop = sr, n_fft, hop
    self.buf = np.zeros(int(sr*seconds), dtype=np.float32)
    self.last_t  = 0.0
    self.prev_ch = None
    self._out = dict(base_hue=200.0, chroma=np.zeros(12), onset=0.0)
  
  def push(self, x):
    n = len(x)
    if n >= len(self.buf): self.buf[:] = x[-len(self.buf):]
    else: self.buf[:-n] = self.buf[n:]; self.buf[-n:] = x
  
  def features(self):
    if (time.monotonic() - self.last_t) < ANALYZE_EVERY_SEC:
      return self._out
    self.last_t = time.monotonic()
  
    y = self.buf
    if not np.any(y): return self._out
    
    y_h, y_p = librosa.effects.hpss(y, margin=1.0)
    
    S_h = np.abs(librosa.stft(y_h, n_fft=self.n_fft,hop_length=self.hop, window="hann"))
    S_p = np.abs(librosa.stft(y_p, n_fft=self.n_fft,hop_length=self.hop, window="hann"))
    
    # Chroma
    C = librosa.feature.chroma_stft(S=S_h, sr=self.sr)
    chroma = C.mean(axis=1)
    if np.sum(chroma) > 1e-9: chroma = chroma / np.sum(chroma)
    
    # top-3 chroma - circular midpoint hue
    PC2HUE = np. array ([120, 150, 180, 200, 220, 250, 280, 300, 0, 30, 60, 90], float)
    top2 = np.argpartition(-chroma,2) [:2]
    top2 = top2[np.argsort(-chroma[top2])]
    h1, h2 = float(PC2HUE[top2[0]]), float(PC2HUE[top2[1]])
    base_hue = circ_lerp(h1, h2, 0.5)
    
    # onsets and bpm
    hop = self.hop
    oenv = librosa.onset.onset_strength(S=S_p, sr=self.sr, hop_length=hop)
    
    self._out = dict(base_hue=base_hue, chroma=chroma, onset=oenv [-1])
    return self._out

# HUD

def hud_init():
    flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE | pygame.FULLSCREEN
    screen = pygame.display.set_mode((W,H), flags)
    font = pygame.font.SysFont("monospace", 20)
    return screen, font

def meter(screen, font, x,y,w,h,val,label):
    pygame.draw.rect(screen,(40,40,40),(x,y,w,h),2)
    fill=int(clamp01(val)*(w-4))
    pygame.draw.rect(screen,(180,180,220),(x+2,y+2,fill,h-4))
    screen.blit(font.render(f"{label}: {val:0.2f}", True, (210,210,210)), (x, y-24))

def slider(screen, font, x,y,w,h,val,label):
    pygame.draw.rect(screen,(60,60,60),(x,y,w,h),0)
    knob=int(clamp01(val)*w)
    pygame.draw.rect(screen,(240,240,240),(x, y, knob, h))
    pygame.draw.rect(screen,(25,25,25),(x,y,w,h),2)
    screen.blit(font.render(label, True, (200,200,200)), (x, y-22))

# MAIN

def main():
  global OUTPUTS_ENABLED, CAMERA_ENABLED, ENABLE_CUBE, BLACKOUT, SHOW_CAMERA_PREVIEW, SHOW_CHROMA
  
  pygame.init()
  screen, font = hud_init()
  
  # Audio in
  stream=None
  if not SAFE_MODE:
    stream = sd.InputStream(callback=_audio_cb, channels=1, samplerate=SR, blocksize=HOP)
    stream.start()
  
  # Camera
  if CAMERA_ENABLED:
    cam_open()
  
  # Analyzer
  AN = LibrosaAnalyzerLite()
  
  # Intensity:
  IC = IntensityController(120.0)
  Ostats = RollingStats(seconds=15.0, fps=30)
  
  # Pattern: base - accl → base → acc2
  
  pat = [0,1,0,2]; pat_idx=0
  hue = 300.0
  comp = 120.0
  sat = 0.95
  val = 0.6
  cube_flip=False
  
  # Smoothing
  Hout=EMA(1); Sout=EMA(0.25); Vout=EMA(1)
  camHue=EMA (0.08); camVal=EMA (0.08)
  Lema=EMA(1); MotEMA=EMA(0.1); IEMA = EMA(0.3)
  
  t_last_update = 0.0
  UPDATE_MIN_DT = 0.15
  
  # draw at most 10 Hz
  
  px_per_edge = (PIXELS_PER_UNI) //12
  edge_idx = np.repeat(np.arange(12), px_per_edge) # (288, )
  
  clock = pygame.time.Clock()
  
  while True:
    t_now = time.monotonic()
    update_hue = False
  
    # events
    for e in pygame.event.get():
    if e.type==pygame.QUIT: return
    if e.type==pygame.KEYDOWN:
    if pygame.K_1 <= e.key <= pygame.K_4:
      scene_idx = e.key - pygame.K_1
      scene_name = scene_names[scene_idx]
      scene_t0 = time.monotonic()
    elif e.key in (pygame.K_ESCAPE, pygame.K_q):
    return
    elif e.key == pygame.K_b:
      BLACKOUT = not BLACKOUT
    if BLACKOUT: par_rgb(0,0,0); cube_push_rgb_bytes(b'\x00\x00\x00'*PIXELS_PER_UNI)
    elif e.key == pygame.K_v:
      OUTPUTS_ENABLED = not OUTPUTS_ENABLED
      par_rgb(0,0,0); cube_push_rgb_bytes(b'\x00\x00\x00'*PIXELS_PER_UNI)
    
    # ---- ingest audio
    x = get_audio_chunk()
    if x is not None: AN.push(x)
    feats = AN.features()
    chroma, onset = feats['chroma'], feats['onset']
    
    # ---- camera
    if CAMERA_ENABLED:
      frame, h_room, v_room, motion = vision_feats()
      h_room = camHue(h_room); v_room = camVal(v_room); motion = MotEMA(motion)
    
    # ---- intensity
    I, Lvl, z = IC.update(onset, Ostats, time.monotonic())
    
    # ---- outputs
    
    if z>0.6 and (t_now - t_last_update > UPDATE_MIN_DT):
    update_hue = True
    
    # sat
    sat = 0.95
    
    # val
    base_v = 0.8 # or your camera-aware value
    gamma = 2 - 2.5*I
    val_p = base_v ** gamma # I=0 → softer; I=1 → punchy
    val_p = Vout(val_p)
    val_c = 0.8
    
    # strobe
    strobe = 0
    if Lvl==3:
      strobe = 160
    
    # hue
    if update_hue:
      hue = feats['base_hue']
      hue = Hout(hue)
      comp = (hue + 90) % 360
      t_last_hue_update = t_now
    
    r,g,b = hsv_to_rgb(hue, sat, val_p)
    cr, cg, cb = hsv_to_rgb(comp, sat, val_c)
    r,g,b = max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))
    cr,cg,cb = max(0,min(255,cr)), max(0,min(255,cg)), max(0,min(255,cb))
    
    par_rgb(r,g,b, strobe=strobe)
    
    # --- HyperCube chroma visualization
    if ENABLE_CUBE and CUBE_OK and (t_now - t_last_update > UPDATE_MIN_DT):
      cube_rgb(cr,cg,cb, IEMA(I))
    
    # ---- HUD
    screen.fill((10,10,25))
    title = pygame.font.SysFont("monospace", 22).render(f"PHOSPHOR", True, (200,220,255))
    screen.blit(title,(40,24))
    meter(screen,font,40,80,  600,24, clamp01(I),   "Intensity")
    meter(screen,font,40,140, 600,24, clamp01(Lvl/3),  "Intensity Level")
    if CAMERA_ENABLED:
      meter(screen,font,40,200, 600,24, camVal.y or 0.0, "Cam Brightness")
      meter(screen,font,40,260, 600,24, (camHue.y or 0.0)%360/360.0, "Cam Hue (norm)")
    slider(screen,font,40,320, 600,16, val, "Out V")
    slider(screen,font,40,360, 600,16, sat, "Out S")
    slider(screen,font,40,400, 600,16, (hue%360)/360.0, "Out H")
    if CAMERA_ENABLED and frame is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h0,w0 = rgb.shape[:2]
        surf = pygame.image.frombuffer(rgb.tobytes(), (w0,h0), "RGB")
        screen.blit(pygame.transform.scale(surf,(3*w0,3*h0)), (40,440))
    
    pygame.display.flip()
    
    clock.tick(30)

# BOOT / CLEANUP

if __name__=="__main__":
    try:
        main()
    finally:
        try: par_rgb(0,0,0)
        except: pass
        try:
            if _cube_sender:
                for u in CUBE_UNIVERSES: _cube_sender.deactivate_output(u)
                _cube_sender.stop()
        except: pass
        try:
            if _cap: _cap.release()
        except: pass
        try: pygame.quit()
        except: pass
