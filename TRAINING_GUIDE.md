# Training Guide & YouTube Video Instructions

Complete guide for training the DQN self-driving car and creating the YouTube demonstration video.

---

## Prerequisites

### Required Software

```bash
# Python 3.8+
python3 --version

# Required packages
pip install torch numpy PyQt6
```

### Verify Installation

```bash
cd ass16
python3 verify_setup.py
```

---

## Running the Training

### Step 1: Launch the Application

```bash
cd ass16
python3 citymap_assignment.py
```

The application window will open with:
- Left panel: Controls, stats, and logs
- Right panel: Map view (map3.jpg loaded by default)

### Step 2: Place the Car (Starting Position)

1. **Left-click** on the map where you want the car to start
2. Choose a location on a white/bright road
3. Avoid placing too close to edges or obstacles
4. **Recommended**: Start in a wide road area for easier initial learning

### Step 3: Place 3 Targets (A1 → A2 → A3)

**Assignment Requirement**: Your map must have 3 targets that the car visits sequentially.

1. **Left-click** to place Target 1 (A1) - Cyan color
2. **Left-click** to place Target 2 (A2) - Magenta color  
3. **Left-click** to place Target 3 (A3) - Green color
4. **Right-click** to confirm target placement

**Target Placement Strategy**:
- **A1 (Easy)**: Place relatively close to start, on a straight road
- **A2 (Medium)**: Place across a curve or requiring a turn
- **A3 (Hard)**: Place across more complex navigation (river, multiple turns)

### Step 4: Start Training

1. Press **SPACE** or click **▶ START** button
2. Training begins automatically
3. Watch the car learn!

### Step 5: Monitor Training Progress

**Left Panel Shows**:
- **Epsilon**: Exploration rate (starts at 1.0, decays to 0.001)
- **Last Reward**: Score from previous episode
- **Reward History Chart**: Visual progress
- **Logs**: Episode outcomes (CRASH/GOAL/ALL TARGETS COMPLETED)

**Indicators of Good Learning**:
- ✅ Epsilon decreasing over time
- ✅ Reward chart trending upward
- ✅ Fewer crashes, more "GOAL" messages
- ✅ Car starts following roads
- ✅ "ALL 3 TARGETS COMPLETED!" messages appearing

### Step 6: Let It Train

**Training Time**: Approximately 15-30 minutes for visible learning

**Milestones to Watch For**:
1. **Early (0-5 min)**: Random movement, many crashes, ε ≈ 0.8-1.0
2. **Mid (5-15 min)**: Starting to follow roads, occasional targets, ε ≈ 0.3-0.5
3. **Late (15-30 min)**: Smooth navigation, completing all 3 targets, ε ≈ 0.01-0.1

---

## Creating the YouTube Video

### What to Record

Your video should demonstrate:

1. **Setup Phase** (30 seconds)
   - Show the map loaded
   - Place car starting position
   - Place 3 targets with explanation

2. **Early Training** (30-60 seconds)
   - Show random initial behavior
   - Many crashes expected
   - High epsilon value visible

3. **Learning Progress** (1-2 minutes)
   - Show improvement over time
   - Reward chart trending up
   - Fewer crashes

4. **Final Performance** (1-2 minutes)
   - Show successful navigation through all 3 targets
   - Smooth driving behavior
   - "ALL 3 TARGETS COMPLETED!" message

5. **Optional: Commentary**
   - Explain what's happening
   - Point out key metrics
   - Discuss parameter choices

### Screen Recording Tools

#### macOS
```bash
# Built-in: QuickTime Player
# 1. Open QuickTime Player
# 2. File → New Screen Recording
# 3. Select recording area
# 4. Click Record

# Alternative: OBS Studio (free)
brew install --cask obs
```

#### Windows
```
# Built-in: Xbox Game Bar
# Press Win + G → Click Record

# Alternative: OBS Studio (free)
# Download from: https://obsproject.com/
```

#### Linux
```bash
# OBS Studio
sudo apt install obs-studio

# Or SimpleScreenRecorder
sudo apt install simplescreenrecorder
```

### Video Recording Tips

1. **Resolution**: Record at 1080p or higher
2. **Frame Rate**: 30 FPS minimum
3. **Audio**: Add voiceover explaining what's happening (optional but recommended)
4. **Length**: 3-5 minutes is ideal
5. **Quality**: Ensure the map and car are clearly visible

### Suggested Video Structure

```
0:00 - 0:30  Introduction & Setup
0:30 - 1:00  Early random behavior
1:00 - 2:30  Learning progression (can speed up)
2:30 - 4:00  Final trained performance
4:00 - 4:30  Summary & conclusion
```

### Video Editing (Optional)

- Speed up slow training sections (2x-4x)
- Add text annotations highlighting key moments
- Include reward chart close-ups

### Uploading to YouTube

1. Go to [YouTube Studio](https://studio.youtube.com/)
2. Click "Create" → "Upload videos"
3. Add title: "DQN Self-Driving Car - Assignment 2 - [Your Name]"
4. Add description explaining the project
5. Set visibility to "Unlisted" or "Public"
6. Copy the video URL for submission

---

## Troubleshooting

### Car Crashes Immediately

**Possible Causes**:
- Car placed on obstacle (dark pixel)
- Speed too high (check SPEED value)

**Solution**: Place car on clearly white/bright road

### Car Doesn't Learn

**Possible Causes**:
- Epsilon too low (check it's starting at 1.0)
- Targets too far apart
- Parameters not fixed

**Solution**: Run `python3 verify_setup.py` to check parameters

### Application Won't Start

**Possible Causes**:
- Missing dependencies
- PyQt6 not installed

**Solution**:
```bash
pip install PyQt6 torch numpy
```

### Map Not Loading

**Possible Causes**:
- map3.jpg not in ass16 folder
- Wrong working directory

**Solution**:
```bash
cd ass16
ls *.jpg  # Should show map files
python3 citymap_assignment.py
```

---

## Alternative Maps

To use a different map, either:

1. **Load via UI**: Click "📂 LOAD MAP" button
2. **Change default**: Edit line 657 in `citymap_assignment.py`:
   ```python
   self.setup_map("map4.jpg")  # Night aerial map
   ```

### Available Maps

| Map | Description | Difficulty |
|-----|-------------|------------|
| map1.jpg | Realistic aerial city | Medium |
| map2.jpg | Grid city with waterways | Easy |
| map3.jpg | Paris-style with river | Medium (default) |
| map4.jpg | Night aerial (stunning visuals) | Medium |
| map5.jpg | River city, organic layout | Medium |

---

## Quick Checklist

Before recording your video, ensure:

- [ ] Application launches without errors
- [ ] Map loads correctly
- [ ] Car can be placed on roads
- [ ] 3 targets can be placed
- [ ] Training starts when pressing SPACE
- [ ] Epsilon decays during training
- [ ] Reward chart updates
- [ ] Car eventually navigates successfully
- [ ] "ALL 3 TARGETS COMPLETED!" appears after training

---

## Support Files

- `verify_setup.py` - Verifies all dependencies and parameters
- `CHANGES.md` - Documents all code modifications
- `ASSIGNMENT_SOLUTION.md` - Answers to theory questions
- `README.md` - General project information

Good luck with your training and video creation! 🚗💨

