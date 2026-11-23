# 🕳️ Wumpus World – Student Edition

Build an AI agent that explores a dangerous cave, finds the gold, and gets out alive, using **only logic**.

---

## 📖 Project Overview

The **Wumpus World** is a classic AI problem. Your agent starts at the bottom-left corner of a **20 × 20 grid** at position **(1, 1)**, facing **East**. Somewhere in the cave there is:

| Hazard / Item | What it does | How your agent can sense it |
|---|---|---|
| 🕳️ **Pits** | Fall in and you die | **Breeze** in the 4 neighbouring cells |
| 👹 **Wumpus** | Walk into one and you die | **Stench** in the 4 neighbouring cells |
| 🏆 **Gold** | The treasure | **Glitter** on the same cell |

Your agent never sees the map. Each turn it receives an *observation* with five percepts:

`breeze` · `stench` · `glitter` · `bump` (walked into a wall, **or** your arrow missed and hit the wall) · `scream` (a Wumpus was killed by your arrow)

…plus its own `position`, `direction`, `arrows_remaining` and `has_gold`.

**Goal:** pick up the gold (`GRAB`) and return to (1, 1) and `CLIMB` out, scoring as high as possible.

### Actions

| Action | Effect |
|---|---|
| `Action.MOVE_FORWARD` | Move one cell in the facing direction |
| `Action.TURN_LEFT` / `Action.TURN_RIGHT` | Rotate 90° |
| `Action.SHOOT` | Fire an arrow straight ahead. You get exactly one arrow per Wumpus on the map, set automatically from Number of Wumpuses |
| `Action.GRAB` | Pick up the gold (only works on the gold's cell) |
| `Action.CLIMB` | Leave the cave (only works at (1, 1)) |

### Scoring

| Event | Points |
|---|---|
| Every action | −1 |
| Shooting (extra) | −10 |
| Grabbing the gold | +1000 |
| Dying | −1000 |
| Forfeit (crash, timeout, too many invalid replies) | −1000 |

> The best agents don't guess. They **reason**: "I felt a breeze at (3,1) and none at (2,1), so a pit might be at (3,2) or (4,1)…" Build a knowledge base and only step on cells you can prove are safe.

---

## 🧰 Prerequisites

1. **Python 3.11 or newer.** Check with:
   ```bash
   python --version
   ```
   (On macOS or Linux you may need `python3`.)
2. **pygame** (draws the board):
   ```bash
   pip install pygame
   ```
3. **Tkinter** (the editor window). It comes with Python on Windows and macOS.
   On Ubuntu/Debian install it with:
   ```bash
   sudo apt install python3-tk
   ```

---

## ▶️ How to Run

1. Download or clone this repository and open a terminal in its folder.
2. Start the app:
   ```bash
   python student_app.py
   ```
3. Two windows open: the **editor** (code box and buttons) and the **board** (the cave).
4. Write or paste your agent into the code box. A working example is already there.
5. Choose the difficulty with **Number of Pits** and **Number of Wumpuses**.
6. Use the buttons:

| Button | What it does |
|---|---|
| **Step** | Plays exactly **one** action. Use it to debug turn by turn. |
| **Run All** | Plays the whole game at high speed. Press **Step** to pause it. |
| **Reset** | Puts the board back to turn 0 on the same map. Press it again on a fresh board for a **new random map**. |

7. Read the status line at the bottom of the editor, and the **trace log** on the right of the board (turn, action, score, percepts).

> 🔍 **Board colours:** the board shows the *true* map (pits, Wumpus, gold) so you can debug. Your agent **cannot** see it. Thin marks show the real breeze/stench; bold marks show what your agent actually perceived.

### Errors

If your code has a **syntax error**, a popup shows the problem, and the faulty line is highlighted in the code box. Runtime errors (an exception in `act()`) show a traceback in a popup. Fix the code and press **Step** or **Run All** again.

---

## 🧩 Agent Format

Your code must define a class named **`StudentAgent`** that inherits from `Agent` and implements `act`:

```python
from wumpus_sdk.actions import Action
from wumpus_sdk.agent import Agent
from wumpus_sdk.dto import ObservationDTO


class StudentAgent(Agent):
    """My Wumpus World agent."""

    def __init__(self) -> None:
        super().__init__()          # always call this first
        self.visited = set()        # your own memory lives here

    def act(self, observation: ObservationDTO) -> Action:
        pos = observation.position            # Position with .x and .y (1..20)
        percept = observation.percept         # .breeze .stench .glitter .scream .bump
        self.visited.add(pos)

        if percept.glitter:
            return Action.GRAB
        if observation.has_gold and (pos.x, pos.y) == (1, 1):
            return Action.CLIMB
        if percept.bump:
            return Action.TURN_LEFT
        return Action.MOVE_FORWARD
```

**What you receive every turn (`ObservationDTO`):**
`step_index`, `position`, `direction`, `arrows_remaining`, `has_gold`, `percept`.

**Optional hooks** you may override:

| Method | Called when |
|---|---|
| `on_game_start(self, session)` | Once, before the first action. `session` has the grid size, `max_actions`, the score rules and `agent_seed`. |
| `on_invalid_action(self, notice)` | Your reply was rejected. You get another try, but too many strikes end the game. |
| `on_game_over(self, result)` | The game ended. |

**Rules and limits**

- `act()` must return an **`Action`** (not a string, not `None`) within **2 seconds**.
- The practice app stops a game after **500 actions**.
- Use `self.rng` (not the `random` module) if you need randomness. It is seeded, so a replay of the same map is identical.
- ⚠️ **Do not use `print()`.** Your agent talks to the game through standard output, and a stray `print()` breaks the game. To debug, print to the error stream instead:
  ```python
  import sys
  print("my debug message", file=sys.stderr)
  ```
- Your agent object lives for one game. Keep everything you learn in `self`.
- Only use the Python standard library.

---

## 🔒 Security Warning: No Cheating Possible

Your agent does **not** run inside the game program. For every game, `student_app.py` saves your code to a temporary file and launches it in a **separate, isolated Python process**. The game and your agent only exchange small JSON messages: observations in, actions out.

This means:

- ❌ You **cannot** read the hidden `World` (pits, Wumpus, gold) with `gc`, `inspect`, `sys._getframe`, `globals()`, or any similar trick. That memory belongs to a different process.
- ❌ You **cannot** change the score or the game state.
- ❌ A crash, an infinite loop, or a reply taking longer than 2 seconds **forfeits the game** (−1000 points). This is exactly how the final grading works.
- ✅ The only information you get is the percepts and your own position. **You must rely on logic and inference.**

Attempts to bypass the isolation, read the instructor's files, or interfere with the grader are a violation of the course's academic integrity policy and will be treated as cheating.

---

## 📤 Submission

Submit **only your agent's `.py` file**.

1. Copy your code from the editor into a file named `firstname_lastname_studentid.py`
   (for example `sara_ahmadi_40212345.py`).
2. Make sure the file:
   - defines `class StudentAgent(Agent):`
   - has all its imports at the top (`wumpus_sdk.actions`, `wumpus_sdk.agent`, `wumpus_sdk.dto`)
   - contains no `print()` calls without `file=sys.stderr`
   - uses only the standard library
3. Test it one last time: paste it into the code box and press **Run All** on a few different maps and difficulty settings.
4. Upload the single `.py` file to the place your instructor tells you (course portal, e-mail, or a pull request).

Do **not** submit the whole repository, screenshots, or modified framework files. Only your agent file is graded, and it runs on the instructor's machine in the same isolated way as here.

---

## 💡 Tips

- Start with **0 pits and 1 Wumpus**, then raise the difficulty.
- Use **Step** and the trace log to see exactly what your agent perceived.
- Track a set of *visited* cells and a set of *provably safe* cells.
- A breeze or stench on the very first cell is a special case. Think about what you can and cannot conclude.
- Killing a Wumpus (`SHOOT` + `scream`) removes its stench from the map.
- After a `SHOOT`, `scream` means a hit and `bump` means the arrow missed and flew into the wall. Check your last action to tell this `bump` from walking into a wall.

Good luck, and may your logic be sound! 🧠