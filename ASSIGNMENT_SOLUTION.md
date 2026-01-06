# Assignment 2: Theory Questions - Solutions

## Question (a): What happens when "boundary-signal" is weak compared to the last reward?

### Understanding the Terms

In this DQN implementation:
- **Boundary-signal** = The crash/collision penalty (currently set to `-100` in line 269)
- **Last reward** = The target reached reward (currently set to `+100` in line 273)

### Answer

When the boundary-signal (crash penalty) is **weak** compared to the target reward, the following problems occur:

#### 1. **Reckless Navigation Behavior**
The agent will take high-risk shortcuts because the potential gain from reaching the target outweighs the fear of crashing.

**Example:**
- If crash penalty = -10 and target reward = +100
- Agent might think: "Even if I crash 5 times, the 6th successful run gives me net positive reward"
- This leads to aggressive, collision-prone driving

#### 2. **Reward Hacking**
The agent optimizes for reaching the target at any cost, ignoring safety:
- Cuts corners too close to walls
- Takes dangerous diagonal paths through narrow spaces
- Doesn't learn to maintain safe distances from obstacles

#### 3. **Unstable Learning**
The Q-values become dominated by the target reward:
- Q(safe_action) ≈ Q(risky_action) because crash penalty is negligible
- Network struggles to differentiate between safe and dangerous states
- Training becomes noisy and unpredictable

#### 4. **Real-World Analogy**
Imagine a delivery driver paid $100 per delivery but only fined $10 for accidents. They would likely drive recklessly because the reward vastly outweighs the penalty.

### Optimal Balance

In our implementation, we use:
- **Crash penalty = -100**
- **Target reward = +100**

This 1:1 ratio ensures the agent:
- Values reaching the target
- But equally fears crashing
- Results in cautious, learned navigation behavior

### Mathematical Perspective

With GAMMA = 0.95, the expected value of a risky path:
```
E[risky] = P(success) × (+100 × 0.95^n) + P(crash) × (-100)
```

For balanced penalties, the agent only takes risks when P(success) is high.

---

## Question (b): What happens when Temperature is reduced?

### Clarification

The term "Temperature" in reinforcement learning typically refers to the **softmax temperature** in action selection. However, this implementation uses **epsilon-greedy** exploration instead of softmax.

In this codebase, the analogous concept is **epsilon (ε)** - the exploration rate.

### Understanding Epsilon (Temperature Equivalent)

- **High epsilon (ε → 1.0)** = High "temperature" = More random actions = More exploration
- **Low epsilon (ε → 0.0)** = Low "temperature" = More greedy actions = More exploitation

### Answer: Effects of Reducing Epsilon/Temperature

#### 1. **Decreased Exploration**
```
Low ε → Agent almost always picks action with highest Q-value
```
- The agent stops trying new actions
- Gets stuck in locally optimal but globally suboptimal strategies
- May never discover better paths that require initial "bad" moves

#### 2. **Faster Convergence (If Already Trained)**
Once the Q-network has learned good values:
- Low epsilon = consistently good performance
- Agent exploits learned knowledge effectively
- Smooth, predictable navigation

#### 3. **Risk of Local Optima (If Reduced Too Early)**
If temperature/epsilon is reduced before sufficient exploration:
- Agent commits to first "okay" strategy found
- May never discover optimal route
- Gets trapped in suboptimal behavior

#### 4. **Greedy Action Selection**
With ε = 0.001 (our minimum):
```
99.9% of the time: action = argmax(Q(s, a))  # Best known action
0.1% of the time: action = random()          # Rare exploration
```

### Temperature Decay in This Implementation

Line 338 shows epsilon decay:
```python
if self.epsilon > 0.001: self.epsilon *= 0.9995
```

This implements **gradual temperature reduction**:
- Starts hot (ε = 1.0): 100% exploration
- Cools down over ~10,000 steps
- Final cold state (ε = 0.001): 0.1% exploration

### Optimal Strategy

The decay schedule balances:
1. **Early training**: High exploration to discover all possible paths
2. **Mid training**: Decreasing exploration as good paths are found
3. **Late training**: Minimal exploration, maximum exploitation

---

## Question (c): What is the effect of reducing gamma (γ)?

### Understanding Gamma

Gamma (γ) is the **discount factor** in the Bellman equation:
```
Q(s, a) = r + γ × max Q(s', a')
```

It determines how much the agent values **future rewards** compared to **immediate rewards**.

### Effects of Reducing Gamma

#### 1. **Short-Sighted Decision Making**

| Gamma | Effective Planning Horizon | Behavior |
|-------|---------------------------|----------|
| 0.99 | ~100 steps | Long-term strategic |
| 0.95 | ~20 steps | Balanced |
| 0.90 | ~10 steps | Medium-term |
| 0.50 | ~2 steps | Very short-sighted |
| 0.01 | Immediate only | Myopic (broken) |

**Formula**: Effective horizon ≈ 1/(1-γ)

#### 2. **Failure to Reach Distant Goals**

With low gamma (e.g., γ = 0.1):
```
Reward 10 steps away is worth: 100 × 0.1^10 = 0.00000001 ≈ 0
```
The agent sees no value in distant targets!

#### 3. **Greedy Immediate Behavior**

Low gamma causes:
- Agent only avoids immediate crashes
- Doesn't plan routes around obstacles
- Takes actions that feel good now but lead to dead ends
- Cannot navigate to targets that require initial detours

#### 4. **Example: Navigation Scenario**

Consider a path where the agent must go around a wall:

**With γ = 0.95 (CORRECT):**
```
Step 1: Go left (away from target) → small negative reward
Step 2: Continue around wall → small negative reward  
Step 3: Clear path to target → future +100 reward visible
Decision: Takes the detour because γ^3 × 100 = 85.7 is significant
```

**With γ = 0.01 (BROKEN):**
```
Step 1: Target is ahead, but wall blocks
Step 2: Going left gives immediate -0.1 reward
Step 3: Future +100 reward is worth: 0.01^3 × 100 = 0.0001 ≈ 0
Decision: Bangs head against wall because can't see value of detour
```

#### 5. **Training Instability**

Low gamma causes:
- Q-values become unstable (only tracking immediate rewards)
- Network can't learn meaningful value estimates
- Training oscillates instead of converging

### Visual Representation

```
Future Reward Decay at Different Gamma Values:

γ = 0.95: ████████████████████░░░░░░░░░ (Plans far ahead)
γ = 0.90: ████████████░░░░░░░░░░░░░░░░░ (Medium planning)
γ = 0.50: ████░░░░░░░░░░░░░░░░░░░░░░░░░ (Very short-sighted)
γ = 0.01: █░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (Only immediate)
         0    5    10   15   20   25   30  (Steps into future)
```

### Conclusion

Reducing gamma makes the agent:
1. **Myopic**: Only considers immediate rewards
2. **Unable to plan**: Cannot execute multi-step strategies
3. **Stuck**: Cannot reach goals requiring detours
4. **Inefficient**: Takes suboptimal paths

**Our fixed value of γ = 0.95** provides:
- ~20 step planning horizon
- Ability to navigate around obstacles
- Balance between immediate safety and long-term goal reaching

---

## Summary Table

| Aspect | Effect When Reduced |
|--------|-------------------|
| **Boundary Signal** | Reckless behavior, ignores safety |
| **Temperature/Epsilon** | Less exploration, risk of local optima |
| **Gamma** | Short-sighted, cannot plan ahead |

All three parameters need careful tuning for successful reinforcement learning!

