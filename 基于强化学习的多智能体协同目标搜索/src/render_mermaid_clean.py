import os
import subprocess
import hashlib
from PIL import Image

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.join(BASE_DIR, "assets", "mermaid")
MMDC_CMD = os.path.normpath(os.path.join(BASE_DIR, "node_modules", ".bin", "mmdc.cmd"))
os.makedirs(ASSETS_DIR, exist_ok=True)

mermaid_sources = {
    "block1_loop": """%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '22px'}}}%%
flowchart LR
    subgraph LOOP["中心化环境交互循环 (每步 Δt=0.5s)"]
        direction LR
        S["全局状态 S_t"] --> O["局部观测 o_i"]
        O --> CONCAT["观测拼接 120-dim"]
        CONCAT --> POLICY["共享策略网络 π_θ"]
        POLICY --> A["联合动作 A_t"]
        A --> WIND["风场叠加"]
        WIND --> P["新位置 P_{t+1}"]
        P --> REWARD["团队共享奖励 R_t"]
        REWARD --> S
    end
    style POLICY fill:#ffe4b5,stroke:#333,stroke-width:2px
    style REWARD fill:#e6ffe6,stroke:#333,stroke-width:2px""",

    "block2_reward": """%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '20px'}}}%%
flowchart TB
    R1["r_find +500"] --- R2["r_signal 20·Δs"] --- R3["r_explore +1/−0.01"] --- R4["r_collision −0.05"]
    R5["r_time −0.02"] --- R6["r_attitude −0.1|ψ̇|"] --- R7["r_energy −0.02|v-W|"]
    R1 --> SUM["Σ → R_total"]
    R2 --> SUM
    R3 --> SUM
    R4 --> SUM
    R5 --> SUM
    R6 --> SUM
    R7 --> SUM
    style R2 fill:#ffe4b5,stroke:#333,stroke-width:2px
    style SUM fill:#e6ffe6,stroke:#333,stroke-width:2px""",

    "block3_train": """%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '22px'}}}%%
flowchart LR
    subgraph TRAIN["参数共享 PPO 训练流程"]
        direction LR
        INIT["初始化 π_θ, V_φ"] --> RESET["重置环境"]
        RESET --> ROLLOUT["Rollout 2048步"]
        ROLLOUT --> GAE["GAE估计"]
        GAE --> UPDATE["策略更新"]
        UPDATE --> VUPDATE["价值更新"]
        VUPDATE --> CKPT{"步数≥500k?"}
        CKPT -->|是| SAVE["保存ckpt"]
        CKPT -->|否| COND{"总步数≥1M?"}
        SAVE --> COND
        COND -->|否| RESET
        COND -->|是| END["训练结束"]
    end
    style INIT fill:#f0f8ff,stroke:#333,stroke-width:2px
    style ROLLOUT fill:#e6ffe6,stroke:#333,stroke-width:2px
    style UPDATE fill:#ffe4b5,stroke:#333,stroke-width:2px
    style END fill:#ffcccc,stroke:#333,stroke-width:2px""",

    "block4_network": """%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '22px'}}}%%
flowchart LR
    subgraph INPUT["输入层"]
        direction TB
        O1["Agent 1 40-dim"]
        O2["Agent 2 40-dim"]
        O3["Agent 3 40-dim"]
    end
    INPUT --> CONCAT["拼接 120-dim"]
    CONCAT --> MLP1["Linear 120→256"]
    MLP1 --> MLP2["Linear 256→256"]
    MLP2 --> ACTOR["策略头 π_θ"]
    MLP2 --> CRITIC["价值头 V_φ"]
    ACTOR --> SAMPLE["重参数化采样"]
    SAMPLE --> RESHAPE["reshape(3,3)"]
    RESHAPE --> OUT1["Agent 1 动作"]
    RESHAPE --> OUT2["Agent 2 动作"]
    RESHAPE --> OUT3["Agent 3 动作"]
    CRITIC --> V["V(s)"]
    style MLP2 fill:#fff0f5,stroke:#333,stroke-width:2px
    style ACTOR fill:#ffe4b5,stroke:#333,stroke-width:2px
    style CRITIC fill:#e6e6fa,stroke:#333,stroke-width:2px
    style RESHAPE fill:#e6ffe6,stroke:#333,stroke-width:2px""",
}

def render(name, content, padding=20):
    h = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    png_path = os.path.join(ASSETS_DIR, f"mermaid_{name}_{h}.png")
    mmd_path = png_path.replace('.png', '.mmd')

    with open(mmd_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Use transparent background, no forced square
    cmd = [MMDC_CMD, "-i", mmd_path, "-o", png_path, "-b", "transparent", "-s", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    os.remove(mmd_path)

    if result.returncode != 0:
        print(f"[ERROR] {name}: {result.stderr}")
        return None

    # Crop transparent edges with padding
    img = Image.open(png_path)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA').split()[-1]
        bbox = alpha.getbbox()
    else:
        bbox = img.getbbox()

    if bbox:
        left, upper, right, lower = bbox
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(img.width, right + padding)
        lower = min(img.height, lower + padding)
        cropped = img.crop((left, upper, right, lower))
        cropped.save(png_path)
        print(f"[OK] {name}: {cropped.size} (cropped)")
    else:
        print(f"[OK] {name}: {img.size} (no crop needed)")

    return png_path

print("=== Rendering mermaid: transparent + auto-crop ===")
results = {}
for name, content in mermaid_sources.items():
    path = render(name, content)
    if path:
        results[name] = path

print("\nResults:", results)
