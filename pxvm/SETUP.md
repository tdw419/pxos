# pxVM Setup Guide

Quick guide to get your self-expanding pixel network running!

---

## Step 1: Install Dependencies

```bash
# Install Python packages
pip install -r pxvm/requirements.txt

# Or install manually
pip install numpy pillow requests
```

---

## Step 2: Install LM Studio

1. Download LM Studio from: https://lmstudio.ai/
2. Install it on your system
3. Launch LM Studio

---

## Step 3: Load a Model in LM Studio

1. Open LM Studio
2. Go to "Discover" or "Models" tab
3. Download a model (suggestions):
   - **Mistral 7B** (good balance)
   - **Llama 2 7B** (reliable)
   - **Phi-2** (fast, lightweight)
   - **CodeLlama** (for code tasks)

4. Wait for download to complete

---

## Step 4: Start LM Studio Server

1. In LM Studio, click "Local Server" tab
2. Select your loaded model
3. Click "Start Server"
4. Server should start on `http://localhost:1234`

**Verify it's running:**
```bash
curl http://localhost:1234/v1/models
```

You should see a JSON response with your model info.

---

## Step 5: Test pxVM

### Option A: Run the Demo

```bash
cd pxos
python3 pxvm/integration/lm_studio_bridge.py --demo
```

This will:
1. Query LLM without context
2. Teach the network about pxOS
3. Query again WITH context
4. Show learning improvement!

### Option B: Interactive Mode

```bash
python3 pxvm/integration/lm_studio_bridge.py --interactive
```

Now you can chat and watch the network grow!

### Option C: Examples

```bash
# Basic example
python3 pxvm/examples/quick_start.py basic

# Teaching example
python3 pxvm/examples/quick_start.py teaching

# Interactive example
python3 pxvm/examples/quick_start.py interactive
```

---

## Step 6: Watch Your Network Grow!

Each time you have a conversation:

```bash
# Network grows
pxvm/networks/learning_network.png: 150 rows
→ [conversation happens]
pxvm/networks/learning_network.png: 175 rows
→ [another conversation]
pxvm/networks/learning_network.png: 205 rows
```

**You can open the PNG to see what it learned!**

---

## Troubleshooting

### "Cannot connect to LM Studio"

**Check:**
- Is LM Studio running?
- Is the server started?
- Is it on port 1234?

**Test:**
```bash
curl http://localhost:1234/v1/models
```

### "ModuleNotFoundError: No module named 'PIL'"

**Fix:**
```bash
pip install pillow
```

### "No pixels generated"

**Check:**
- Font installation: `ls /usr/share/fonts/truetype/dejavu/`
- If missing on Linux: `sudo apt install fonts-dejavu`
- If missing on Mac: Fonts should be included
- If missing on Windows: Will use default font

### Network not saving

**Check:**
- File permissions: `ls -la pxvm/networks/`
- Disk space: `df -h`
- Path exists: `ls pxvm/networks/`

---

## Quick Reference

### Start Interactive Mode
```bash
python3 pxvm/integration/lm_studio_bridge.py --interactive
```

### Run Demo
```bash
python3 pxvm/integration/lm_studio_bridge.py --demo
```

### Custom Network Path
```bash
python3 pxvm/integration/lm_studio_bridge.py \
    --network my_network.png \
    --interactive
```

### Custom LM Studio URL
```bash
python3 pxvm/integration/lm_studio_bridge.py \
    --url http://localhost:8080/v1 \
    --interactive
```

---

## Verification Checklist

- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] LM Studio downloaded and running
- [ ] Model loaded in LM Studio
- [ ] Server started on localhost:1234
- [ ] Verified with: `curl http://localhost:1234/v1/models`
- [ ] Run demo successfully: `python3 pxvm/integration/lm_studio_bridge.py --demo`
- [ ] Network PNG created in `pxvm/networks/`

---

## Next Steps

1. **Experiment** with different queries
2. **Watch** the network grow
3. **Open** the PNG to see stored knowledge
4. **Share** your trained networks with others!

---

## System Requirements

**Minimum:**
- Python 3.7+
- 4GB RAM (for small models)
- 5GB disk space (for model + network)

**Recommended:**
- Python 3.9+
- 8GB+ RAM (for better models)
- 10GB+ disk space
- GPU (optional, for faster LM Studio inference)

---

## Platform Notes

### Linux
```bash
# Install system dependencies
sudo apt install python3-pip fonts-dejavu

# Install Python packages
pip3 install -r pxvm/requirements.txt
```

### macOS
```bash
# Install Python packages
pip3 install -r pxvm/requirements.txt
```

### Windows
```powershell
# Install Python packages
pip install -r pxvm/requirements.txt
```

---

## Ready to Go!

```bash
cd pxos
python3 pxvm/integration/lm_studio_bridge.py --interactive
```

**Your self-expanding learning loop starts now!** 🚀
