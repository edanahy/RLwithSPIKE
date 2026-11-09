# Reinforcement Learning with LEGO SPIKE Prime

Two hands-on reinforcement learning activities for LEGO Education SPIKE Prime, plus a live web dashboard.

In both activities, a robot uses **Q-learning** to learn by trial and error. It tries actions, earns rewards, and builds a table of what works. Students can watch every decision as it happens.

**Live dashboard:** <https://edanahy.github.io/RLwithSPIKE/>

Open it in Chrome or Edge to watch your robot learn. You can also click **Try demo** to see a simulated robot with no hardware.

## The activities

### 1D Color Search
A color sensor on a swinging arm scans a row of colored LEGO bricks. The robot's job is to find the **goal** brick, such as green, without running into the **boundary** bricks at either end, such as black.

- **State:** the color under the sensor.
- **Action:** sweep left or sweep right to the next brick.
- **Reward:** +10 for the goal, −10 for a boundary, −1 for every other move. The −1 makes shorter paths score better.

You don't program the track into the robot. It **discovers** each color as it explores, and adds a new row to its Q-table each time. After it has learned the track, rearrange the bricks and watch it re-learn.

**Hardware:** arm motor in port B, color sensor in port D.

### Smart Walker
A "silly walker" flops forward on two motorized legs, and it usually curves off to one side. The Smart Walker learns how fast to run each leg so that it walks straight.

- **State:** which way the robot is pointing compared to its start, sorted into heading zones. The hub's gyro measures this.
- **Action:** a pair of leg speeds (left, right).
- **Reward:** high when pointing straight, negative when it turns away.

The robot learns from what actually happens, so the program works for any leg design.

**Hardware:** left leg motor in port B, right leg motor in port A. The hub can be mounted in any orientation.

## Files

| File | What it is |
|---|---|
| [`rl_color_search.py`](rl_color_search.py) | The 1D Color Search program for the hub |
| [`rl_smart_walker.py`](rl_smart_walker.py) | The Smart Walker program for the hub |
| [`read_color.py`](read_color.py) | Setup checker for Color Search. It shows what the sensor reads for every brick. Run this first. |
| [`index.html`](index.html) | The web dashboard (one self-contained page that also works offline) |

## Getting started

1. **Build the robot** and plug the motors and sensor into the ports listed above.
2. **Color Search only:** run `read_color.py` first. Hold the hub's LEFT button, then RIGHT, to sweep the arm across the track. Check that every brick reads as the right color and that the gaps between bricks read as `NONE`.
3. **Open the program** (`rl_color_search.py` or `rl_smart_walker.py`) in the [SPIKE App](https://spike.legoeducation.com/), as a Python project. Check the **SETTINGS** section at the top of the file, then download the program to a slot on the hub.
4. **Close the SPIKE App**, or disconnect it from the hub. Only one app can talk to the hub at a time.
5. **Start the program on the hub.** Pick its slot with the left/right buttons, then press the center button.
6. **Open the dashboard** and click **USB** or **Bluetooth**. For Bluetooth, press the hub's Bluetooth button first so it blinks.
7. **Train and test** with the hub's buttons:
   - **RIGHT:** train one episode. The robot explores and learns.
   - **LEFT:** test. The robot uses only what it has learned: no random moves and no learning.
   - **Center:** end the program. This erases what it learned.

Messages starting with `>>` tell you what to do next. They appear on the dashboard and in the SPIKE App console.

## About the dashboard

- **It only listens.** The page reads what the hub program prints and never sends commands. Everything is controlled with the hub's buttons, and the robots work without the dashboard. You can connect or reconnect at any time, and the page catches up at the start of the next run.
- **Browser:** Chrome or Edge on a laptop or desktop. Safari and Firefox don't support Web Serial or Web Bluetooth.
- **What it shows:**
  - **Color Search:** the arm on the track, the Q-table filling in, a map of the track rebuilt from the robot's moves, and steps per episode.
  - **Smart Walker:** a live heading dial, the walking path, rewards per episode, and a Q-table heatmap.
- **Save log / Open log…** saves everything the hub printed, so you can replay a session later.
- **Try demo** runs a simulated robot that uses the same learning program. It works without any hardware.
- To use the dashboard offline, download `index.html` and open it from your computer.

## Customizing

Every setting students might change is in the **SETTINGS** section at the top of each program, with comments. That includes ports, motor directions, goal and boundary colors, rewards, leg speeds, heading zones, episode lengths, and the learning parameters:

- **α (alpha):** the learning rate.
- **γ (gamma):** the discount on future rewards.
- **ε (epsilon):** the exploration rate.

The dashboard reads the settings from the hub, so it stays in sync when you change them.

### Data line format
Lines that start with `@` carry data for the dashboard, in the form `@TYPE key=value key=value`. For example:

```
@STEP run=4 t=2 state=BLUE action=RIGHT reward=-1 old=0.00 best_next=0.00 new=-0.50
```

The hub cuts off printed lines at about 255 characters, so keep each `@` line short. That's why large Q-tables are sent one row per line (`@QROW`). You can add your own `print()` lines freely: the dashboard shows them in its console.

## Credits

Developed at the [Tufts Center for Engineering Education and Outreach (CEEO)](https://ceeo.tufts.edu/).

- 1D Color Search: first developed by Tanushree Burman, 2024.
- Smart Walker: first developed by Ziyi Zhang, around 2021.
- Both were adapted for the ML workshop in October 2024. In 2026 they were cleaned up and connected to the web dashboard.

LEGO® and SPIKE™ are trademarks of the LEGO Group, which does not sponsor or endorse this project.
