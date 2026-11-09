# =====================================================================
#  SMART WALKER  -  Reinforcement Learning on LEGO SPIKE Prime
# =====================================================================
#  A "silly walker" has two motors with legs (beams) attached. When the
#  motors spin, the robot flops forward - but it usually curves off to
#  one side. A Smart Walker LEARNS how fast to run each leg so that it
#  walks STRAIGHT, using Q-learning:
#    - STATE  = which way the robot is pointing compared to where it
#               started, measured by the hub's gyro (the IMU), sorted
#               into a few zones: way left ... straight ... way right
#    - ACTION = a pair of leg speeds (left speed, right speed)
#    - REWARD = big when pointing straight, negative when it turns away
#  Because the robot learns from what actually happens, the same program
#  works for ANY leg design: change the legs, train again, and the new
#  robot learns to walk straight too.
#
#  First developed by Ziyi Zhang, Tufts CEEO, circa 2021
#  Adapted for the ML workshop, October 21, 2024
#  Cleaned up and connected to the web dashboard (index.html), 2026
#
#  HARDWARE
#    Left leg motor .... Port B
#    Right leg motor ... Port A
#    The hub can be mounted any way you like (flat, on its side...).
#
#  BEFORE YOU START THE PROGRAM
#    Put both legs in the starting pose you want. After every run the
#    robot puts its legs back in this pose, so every run starts the same.
#
#  BUTTONS (while the program is running)
#    RIGHT button ...... TRAIN one episode (explores and learns)
#    LEFT button ....... TEST: use the best actions it has learned
#                        (no random moves, no learning)
#    Either button during a run stops that run early.
#    You can also stop a run by picking the robot up and turning it around.
#    Center button ..... ends the program (this erases what it learned!)
#
#  Before each run, point the robot at a target across the room (about
#  1.5 meters / 5 feet away) and set it down. "Straight" means the
#  direction it is facing when the run starts.
#
#  The light matrix shows how many training episodes are done. During a
#  run it shows the two leg speeds as bars (a dot in the middle means the
#  action was a random, exploring one). The center button light shows how
#  straight the robot is: green = straight, yellow, orange, red = way off.
#
#  WEB DASHBOARD (optional)
#    Open index.html in Chrome or Edge and connect over USB or Bluetooth
#    to watch the heading, the Q-table and the learning progress live.
#    Lines starting with ">>" tell you what to do next.
#    Lines starting with "@" are data for the dashboard.
# =====================================================================

from hub import port, button, light, light_matrix, motion_sensor, sound
import color
import motor
import random
import runloop
import time


# =====================================================================
#  SETTINGS - change these to match your robot and to experiment
# =====================================================================

# ---- Ports and motor directions ----
LEFT_MOTOR_PORT = port.B
RIGHT_MOTOR_PORT = port.A
LEFT_MOTOR_DIRECTION = 1     # change to -1 if the left leg spins the wrong way
RIGHT_MOTOR_DIRECTION = -1   # motors on opposite sides face opposite ways,
                             # so one of the two is usually -1

# ---- Actions: leg speeds ----
# Speeds in degrees per second (SPIKE Prime motors go up to about 1000).
# Every combination of a left speed and a right speed is one ACTION
# (except both legs stopped): 3 speeds -> 3 x 3 - 1 = 8 actions.
# Spread the speeds out: if they are too close together, every action
# does almost the same thing and there is little to learn.
# More speeds = more actions = a bigger Q-table and longer training!
LEG_SPEEDS = [0, 300, 750]

# ---- States: heading zones ----
# Zone edges in degrees away from straight. [5, 20, 40] makes 7 states:
#   0 = within 5 degrees of straight
#   1 / -1 = 5 to 20 degrees off      2 / -2 = 20 to 40 degrees off
#   3 / -3 = more than 40 degrees off
STATE_EDGES = [5, 20, 40]
# Reward for each zone: straight first, then each zone further away.
# (Needs one more number than STATE_EDGES.)
STATE_REWARDS = [10, -2, -10, -25]

# ---- Runs ----
STEP_TIME_MS = 750     # how long each action runs before the next decision
                       # (long enough for the legs to really change the direction)
TRAIN_STEPS = 20       # steps in one training episode (20 x 0.75 s = 15 seconds)
TEST_STEPS = 40        # steps in one test run (40 x 0.75 s = 30 seconds)
GIVE_UP_ANGLE = 140    # stop the run if the robot turns more than this many degrees
START_DELAY_MS = 1000  # pause after the button press so you can take your hand away

# ---- Learning settings ----
ALPHA = 0.3            # learning rate (0 to 1): how much one new experience changes a Q-value
GAMMA = 0.9            # discount (0 to 1): how much future rewards count compared to right now
EPSILON_START = 0.9    # exploration at the start: 0.9 = 90% random actions
EPSILON_MIN = 0.1      # exploration never drops below this while training
EPSILON_DECAY = 0.008  # exploration drops by this much after every training step
                       # (0.9 -> 0.1 takes 100 steps = 5 training episodes)

# ---- Legs ----
RESET_LEGS_AFTER_RUN = True  # put the legs back in the starting pose after each run
LEG_RESET_SPEED = 200

# ---- Hub mounting ----
AUTO_HUB_ORIENTATION = True  # measure the heading correctly however the hub is mounted
                             # (uses whichever side of the hub is facing up at the start of a run)

# ---- Sound and printing ----
VOLUME = 50                   # hub speaker volume (0 to 100); 0 = silent
PRINT_POLICY_AFTER_RUN = True # print the best action for each state after every run
DASHBOARD_LINES = True        # print the "@..." data lines that the web dashboard needs.
                              # Set to False for a tidier console without the dashboard.


# =====================================================================
#  You do not need to change anything below to get started,
#  but reading it is the best way to see how Q-learning works!
# =====================================================================

# All the actions: (left speed, right speed), skipping "both legs stopped"
ACTIONS = []
for left_speed in LEG_SPEEDS:
    for right_speed in LEG_SPEEDS:
        if left_speed != 0 or right_speed != 0:
            ACTIONS.append((left_speed, right_speed))

# All the states: -3, -2, -1, 0, 1, 2, 3 (with 3 edges)
STATES = list(range(-len(STATE_EDGES), len(STATE_EDGES) + 1))

ZONE_LIGHTS = [color.GREEN, color.YELLOW, color.ORANGE, color.RED]


def num(x, places=2):
    """A number as text with fixed decimals: -0.5 -> '-0.50'."""
    return ("{:." + str(places) + "f}").format(x)


def num_list(values):
    """Numbers as text with no spaces: [1, -0.5] -> '[1.00,-0.50]'."""
    return "[" + ",".join(num(v) for v in values) + "]"


def int_list(values):
    return "[" + ",".join(str(v) for v in values) + "]"


def signed(n):
    """An integer with a + sign if positive: 10 -> '+10'."""
    if n > 0:
        return "+" + str(n)
    return str(n)


def data(line):
    """Print a data line for the web dashboard.
    Keep these lines short (under about 200 characters): the hub can cut off
    long printed lines before they reach the computer."""
    if DASHBOARD_LINES:
        print("@" + line)


def tell(text):
    """Tell the student what to do next (the dashboard shows this in big letters)."""
    print(">> " + text)


# =====================================================================
#  THE AGENT - the "brain" that learns
#  The agent only sees numbers: state 0, 1, 2, ... and action 0, 1, ...
#  It has no idea it is a walking robot.
#  This class is exactly the same in the 1D Color Search program!
# =====================================================================
class QAgent:
    def __init__(self, num_states, num_actions, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions
        # The Q-table: one row per state, one column per action.
        # Everything starts at 0 because the agent knows nothing yet.
        self.q = [[0.0] * num_actions for _ in range(num_states)]

    def add_state(self):
        """Add a row of zeros for a state the agent has just discovered."""
        self.q.append([0.0] * self.num_actions)

    def best_action(self, state):
        """The action with the highest Q-value in this state.
        If several actions tie for best, pick one of them at random."""
        row = self.q[state]
        best = max(row)
        ties = [a for a in range(len(row)) if row[a] == best]
        return random.choice(ties)

    def choose_action(self, state, epsilon):
        """Epsilon-greedy: usually EXPLOIT (pick the best known action),
        but with probability epsilon EXPLORE (pick a random action).
        Returns (action, explored)."""
        if random.random() < epsilon:
            return random.randint(0, len(self.q[state]) - 1), True
        return self.best_action(state), False

    def learn(self, state, action, reward, next_state, done):
        """The Q-learning update rule:

            new Q = old Q + alpha * (reward + gamma * best_next - old Q)

        best_next is the best Q-value in the state we landed in: how good
        things look from here on. If the episode is over there is no future,
        so best_next is 0. Returns (old Q, best_next, new Q)."""
        old = self.q[state][action]
        if done:
            best_next = 0.0
        else:
            best_next = max(self.q[next_state])
        target = reward + self.gamma * best_next
        new = old + self.alpha * (target - old)
        self.q[state][action] = new
        return old, best_next, new


# =====================================================================
#  THE ENVIRONMENT - the robot's world: legs, gyro, rewards
# =====================================================================
async def reset_heading():
    """Make the direction the robot is facing right now count as 'straight' (0)."""
    if AUTO_HUB_ORIENTATION:
        motion_sensor.set_yaw_face(motion_sensor.up_face())
        await runloop.sleep_ms(100)
    motion_sensor.reset_yaw(0)
    await runloop.sleep_ms(50)


def read_heading():
    """How far the robot has turned since the start of the run, in degrees.
    (The hub reports tenths of a degree, so divide by 10.)"""
    return motion_sensor.tilt_angles()[0] / 10


def heading_to_state(heading):
    """Sort a heading into a zone: 0 = straight, bigger numbers = further off."""
    zone = 0
    while zone < len(STATE_EDGES) and abs(heading) > STATE_EDGES[zone]:
        zone += 1
    if heading < 0:
        return -zone
    return zone


def state_number(state):
    """The row of the Q-table for a state: state -3 is row 0, state 3 is row 6."""
    return state + len(STATE_EDGES)


def reward_for(state):
    return STATE_REWARDS[abs(state)]


def state_label(state):
    """A description of a state, like 'turned +5 to +20 deg'."""
    zone = abs(state)
    if zone == 0:
        return "straight (within " + str(STATE_EDGES[0]) + " deg)"
    if state > 0:
        sign = "+"
    else:
        sign = "-"
    low = sign + str(STATE_EDGES[zone - 1])
    if zone == len(STATE_EDGES):
        return "turned more than " + low + " deg"
    return "turned " + low + " to " + sign + str(STATE_EDGES[zone]) + " deg"


def drive(left_speed, right_speed):
    motor.run(LEFT_MOTOR_PORT, LEFT_MOTOR_DIRECTION * left_speed)
    motor.run(RIGHT_MOTOR_PORT, RIGHT_MOTOR_DIRECTION * right_speed)


def stop_legs():
    motor.stop(LEFT_MOTOR_PORT, stop=motor.COAST)
    motor.stop(RIGHT_MOTOR_PORT, stop=motor.COAST)


async def legs_to_start_pose(pose):
    """Turn both legs back to the pose they had when the program started."""
    if not RESET_LEGS_AFTER_RUN:
        return
    motor.run_to_absolute_position(LEFT_MOTOR_PORT, pose[0], LEG_RESET_SPEED)
    await motor.run_to_absolute_position(RIGHT_MOTOR_PORT, pose[1], LEG_RESET_SPEED)
    await runloop.sleep_ms(300)
    stop_legs()


# =====================================================================
#  HUB HELPERS - buttons, light matrix and sounds
# =====================================================================
def stop_requested():
    return button.pressed(button.LEFT) or button.pressed(button.RIGHT)


async def wait_for_release():
    while stop_requested():
        await runloop.sleep_ms(20)


async def wait_for_button(episodes):
    """Show the number of training episodes and wait for LEFT or RIGHT.
    Returns button.LEFT or button.RIGHT."""
    await wait_for_release()
    light_matrix.write(str(episodes))
    shown_at = time.ticks_ms()
    while not stop_requested():
        # Numbers with 2+ digits scroll past, so show them again every few seconds
        if episodes >= 10 and time.ticks_diff(time.ticks_ms(), shown_at) > 3000:
            light_matrix.write(str(episodes))
            shown_at = time.ticks_ms()
        await runloop.sleep_ms(20)
    if button.pressed(button.RIGHT):
        which = button.RIGHT
    else:
        which = button.LEFT
    await wait_for_release()
    return which


async def wait_and_watch(ms):
    """Wait, but return True right away if a button is pressed (stop the run)."""
    end = time.ticks_add(time.ticks_ms(), ms)
    while time.ticks_diff(end, time.ticks_ms()) > 0:
        if stop_requested():
            return True
        await runloop.sleep_ms(20)
    return False


async def beep(frequency, ms):
    if VOLUME > 0:
        await sound.beep(frequency, ms, VOLUME)


def bar_height(speed):
    """0 to 5 lit pixels for a leg speed (slowest = 0, fastest = 5)."""
    level = LEG_SPEEDS.index(speed)
    return int(5 * level / (len(LEG_SPEEDS) - 1) + 0.5)


def show_legs(left_speed, right_speed, explored):
    """Draw the two leg speeds on the light matrix as bars: left bar = left leg."""
    pixels = [0] * 25
    left_bar = bar_height(left_speed)
    right_bar = bar_height(right_speed)
    for row in range(5):
        height_here = 5 - row   # the bottom row is height 1
        if left_bar >= height_here:
            pixels[row * 5 + 0] = 100
            pixels[row * 5 + 1] = 100
        if right_bar >= height_here:
            pixels[row * 5 + 3] = 100
            pixels[row * 5 + 4] = 100
    if explored:
        pixels[2 * 5 + 2] = 100   # a dot in the middle = random (exploring) action
    light_matrix.show(pixels)


def show_zone_light(state):
    light.color(light.POWER, ZONE_LIGHTS[min(abs(state), len(ZONE_LIGHTS) - 1)])


# =====================================================================
#  PRINTING - for people (the console) and for the web dashboard
# =====================================================================
def print_welcome():
    print("")
    print("=================================================")
    print("  SMART WALKER with Reinforcement Learning")
    print("=================================================")
    print("Goal: learn which leg speeds keep the robot walking STRAIGHT.")
    print("States: " + str(len(STATES)) + " heading zones, from "
          + str(STATES[0]) + " to " + signed(STATES[-1]) + " (0 = straight)")
    print("Actions: " + str(len(ACTIONS)) + " (left speed, right speed) combinations from "
          + int_list(LEG_SPEEDS))
    print("Q-table: " + str(len(STATES)) + " x " + str(len(ACTIONS)) + " = "
          + str(len(STATES) * len(ACTIONS)) + " values, all starting at 0")
    print("Rewards by zone (straight first): " + " ".join(signed(r) for r in STATE_REWARDS))
    print("Learning: alpha " + num(ALPHA) + ", gamma " + num(GAMMA) + ", exploration "
          + num(EPSILON_START) + " dropping to " + num(EPSILON_MIN))
    print("")
    print("  RIGHT button = TRAIN one episode (" + str(TRAIN_STEPS) + " steps, explore + learn)")
    print("  LEFT button  = TEST (" + str(TEST_STEPS) + " steps, best actions only, no learning)")
    print("  Press either button during a run to stop it.")
    print("")


def send_config():
    """Describe the activity to the web dashboard, in two short lines.
    (The dashboard rebuilds the list of actions from the speeds, the same way
    this program does.)"""
    data("CONFIG app=walker"
         + " states=" + int_list(STATES)
         + " edges=" + int_list(STATE_EDGES)
         + " rewards=" + int_list([reward_for(s) for s in STATES])
         + " speeds=" + int_list(LEG_SPEEDS))
    data("CONFIG app=walker"
         + " alpha=" + num(ALPHA) + " gamma=" + num(GAMMA)
         + " epsilon_start=" + num(EPSILON_START) + " epsilon_min=" + num(EPSILON_MIN)
         + " epsilon_decay=" + num(EPSILON_DECAY, 4)
         + " train_steps=" + str(TRAIN_STEPS) + " test_steps=" + str(TEST_STEPS)
         + " step_ms=" + str(STEP_TIME_MS) + " give_up=" + str(GIVE_UP_ANGLE))


def send_qtable(agent, episodes, epsilon):
    """Send the whole Q-table to the dashboard: a short header line, then
    each row in pieces of up to 12 values (so no line gets too long)."""
    data("QTABLE app=walker ep=" + str(episodes) + " epsilon=" + num(epsilon)
         + " rows=" + str(len(STATES)))
    for state in STATES:
        row = agent.q[state_number(state)]
        for start in range(0, len(row), 12):
            data("QROW app=walker state=" + str(state) + " from=" + str(start)
                 + " q=" + num_list(row[start:start + 12]))


def print_policy(agent, episodes):
    """For each state, print the action with the highest Q-value."""
    print("What the walker has learned after " + str(episodes) + " training episode(s):")
    for state in STATES:
        row = agent.q[state_number(state)]
        label = "  state {:>2}  {:<28}".format(signed(state), state_label(state))
        if max(row) == 0 and min(row) == 0:
            print(label + " not tried yet")
        else:
            best = row.index(max(row))
            left_speed, right_speed = ACTIONS[best]
            print(label + " best: left " + str(left_speed) + ", right " + str(right_speed)
                  + "   (Q = " + num(row[best]) + ")")
    print("")


# =====================================================================
#  ONE RUN - a training episode or a test
# =====================================================================
async def run_walk(agent, training, episodes, run_number, epsilon):
    """Walk for a number of steps. Returns the new exploration rate."""
    if training:
        mode = "train"
        max_steps = TRAIN_STEPS
        print("=== TRAINING episode " + str(episodes) + " (exploration "
              + num(epsilon) + ") ===")
    else:
        mode = "test"
        max_steps = TEST_STEPS
        print("=== TEST run (after " + str(episodes) + " training episodes) ===")
    print("Starting in " + str(START_DELAY_MS / 1000) + " seconds - hands off!")
    await beep(880, 100)
    await runloop.sleep_ms(START_DELAY_MS)
    await reset_heading()
    await beep(1320, 100)

    heading = read_heading()
    if training:
        epsilon_now = epsilon
    else:
        epsilon_now = 0.0
    data("START app=walker mode=" + mode + " run=" + str(run_number) + " ep=" + str(episodes)
         + " epsilon=" + num(epsilon_now) + " heading=" + num(heading, 1))

    total = 0
    steps = 0
    result = "done"
    for t in range(1, max_steps + 1):
        # 1. Look at the state and choose an action
        state = heading_to_state(heading)
        s = state_number(state)
        if training:
            epsilon_now = epsilon
        action, explored = agent.choose_action(s, epsilon_now)
        left_speed, right_speed = ACTIONS[action]

        # 2. Do the action: run the legs for one step
        drive(left_speed, right_speed)
        show_legs(left_speed, right_speed, explored)
        if await wait_and_watch(STEP_TIME_MS):
            result = "stopped"
            break

        # 3. See where we ended up and get the reward
        new_heading = read_heading()
        if abs(new_heading) > GIVE_UP_ANGLE:
            # Turned too far (or someone turned it around to stop it).
            # We don't learn from this step - a person may have moved the robot.
            result = "turned"
            break
        new_state = heading_to_state(new_heading)
        reward = reward_for(new_state)
        steps = t
        total += reward
        show_zone_light(new_state)

        if explored:
            explore_word = "yes"
            how = "random"
        elif agent.q[s].count(agent.q[s][action]) > 1:
            explore_word = "no"
            how = "tie, random"
        else:
            explore_word = "no"
            how = "best"
        story = ("  step {:>2}: heading {:>4} deg (state {:>2}) -> legs L{} R{} ({}) -> "
                 "heading {:>4} deg (state {:>2}), reward {}").format(
                     t, int(heading), signed(state), left_speed, right_speed, how,
                     int(new_heading), signed(new_state), signed(reward))
        line = ("STEP run=" + str(run_number) + " t=" + str(t)
                + " heading=" + num(heading, 1) + " state=" + str(state)
                + " action=" + str(action) + " left=" + str(left_speed)
                + " right=" + str(right_speed) + " explore=" + explore_word
                + " next_heading=" + num(new_heading, 1) + " next=" + str(new_state)
                + " reward=" + str(reward))

        # 4. Learn from it (only while training)
        if training:
            # The walk has no "finish line", so done is always False here:
            # the future always counts.
            old, best_next, new = agent.learn(s, action, reward, state_number(new_state), False)
            epsilon = max(EPSILON_MIN, epsilon - EPSILON_DECAY)
            line += (" old=" + num(old) + " best_next=" + num(best_next) + " new=" + num(new)
                     + " epsilon=" + num(epsilon))
        print(story)
        data(line)
        heading = new_heading

    stop_legs()

    if steps > 0:
        average = total / steps
    else:
        average = 0
    if result == "done":
        print("Finished " + str(steps) + " steps. Total reward " + signed(total)
              + ", average " + num(average) + " per step (" + str(STATE_REWARDS[0])
              + " = perfectly straight).")
    elif result == "stopped":
        print("Run stopped by a button press after " + str(steps) + " steps.")
    else:
        print("Turned more than " + str(GIVE_UP_ANGLE) + " degrees - run stopped after "
              + str(steps) + " steps.")
    data("END app=walker mode=" + mode + " run=" + str(run_number) + " ep=" + str(episodes)
         + " steps=" + str(steps) + " total=" + str(total) + " average=" + num(average)
         + " result=" + result)

    if result == "done" and average >= STATE_REWARDS[0] / 2:
        light_matrix.show_image(light_matrix.IMAGE_HAPPY)
    else:
        light_matrix.show_image(light_matrix.IMAGE_MEH)
    await beep(660, 150)
    return epsilon


# =====================================================================
#  MAIN PROGRAM
# =====================================================================
async def main():
    print_welcome()
    if len(STATE_REWARDS) != len(STATE_EDGES) + 1:
        print("SETUP PROBLEM: STATE_REWARDS needs exactly one more number than STATE_EDGES.")
        tell("Fix the settings at the top of the program and download it again.")
        light_matrix.show_image(light_matrix.IMAGE_NO)
        return

    agent = QAgent(len(STATES), len(ACTIONS), ALPHA, GAMMA)
    epsilon = EPSILON_START
    episodes = 0   # training episodes so far
    runs = 0       # all runs so far (training + test)
    start_pose = (motor.absolute_position(LEFT_MOTOR_PORT),
                  motor.absolute_position(RIGHT_MOTOR_PORT))
    data("HELLO app=walker version=1")   # tells the dashboard a new session started
    send_config()
    send_qtable(agent, episodes, epsilon)

    while True:
        tell("Point the robot at your target and set it down. "
             "Then press RIGHT to TRAIN or LEFT to TEST.")
        which = await wait_for_button(episodes)
        training = (which == button.RIGHT)

        send_config()   # repeated so a dashboard that connected late can catch up
        send_qtable(agent, episodes, epsilon)
        runs += 1
        if training:
            episodes += 1
        epsilon = await run_walk(agent, training, episodes, runs, epsilon)
        await wait_for_release()
        await legs_to_start_pose(start_pose)

        send_qtable(agent, episodes, epsilon)
        if PRINT_POLICY_AFTER_RUN:
            print_policy(agent, episodes)


runloop.run(main())
