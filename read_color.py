# =====================================================================
#  READ COLOR  -  check your 1D Color Search setup
# =====================================================================
#  Run this BEFORE rl_color_search.py to make sure the color sensor reads
#  every brick correctly. It uses the same rule as the learning program
#  ("the same color several readings in a row = a brick"), so what you
#  see here is what the robot will see.
#
#  HOW TO USE
#    - Turn the arm slowly BY HAND across the bricks, or
#    - HOLD the LEFT or RIGHT button to sweep the arm with the motor,
#      at the same speed the learning program uses.
#      The arm stops by itself when it reaches a boundary brick.
#  Every time the sensor settles on a new color, the hub prints it,
#  shows its letter on the light matrix and lights the center button.
#
#  Hold LEFT until the arm stops at the left end, then hold RIGHT until
#  it stops at the right end. You should see your whole track, e.g.
#      Bricks in order: BLACK MAGENTA BLUE AZURE GREEN YELLOW RED BLACK
#
#  CHECKLIST
#    1. Every brick shows up once, with the right color name.
#    2. Gaps between bricks read as NONE (no color), not as a color.
#       If a gap reads as a color (e.g. WHITE from the table), add that
#       color to IGNORE_COLORS here AND in rl_color_search.py.
#    3. Holding LEFT moves the arm to the LEFT. If not, set
#       MOTOR_DIRECTION = -1 here AND in rl_color_search.py.
#    4. Every brick is a different color, except the two boundary bricks.
#  Wrong or flickering colors? Try: steadier light (no shadows or direct
#  sun), the sensor closer to the bricks (about 1 cm), bricks centered
#  under the sensor's path, or a slower SWEEP_SPEED.
#
#  HARDWARE: Arm motor in Port B, Color Sensor in Port D.
# =====================================================================

from hub import port, button, light, light_matrix
import color
import color_sensor
import motor
import runloop


# =====================================================================
#  SETTINGS - copy these from rl_color_search.py so both match
# =====================================================================
ARM_MOTOR_PORT = port.B
COLOR_SENSOR_PORT = port.D
SWEEP_SPEED = 30          # degrees per second
MOTOR_DIRECTION = 1       # change to -1 if holding LEFT moves the arm to the right
CONFIRM_READINGS = 4      # the same color this many times in a row = a brick
READING_GAP_MS = 25       # milliseconds between color readings
IGNORE_COLORS = []        # colors that are NOT bricks, e.g. [color.WHITE]
BOUNDARY_COLOR = color.BLACK

SHOW_DETAILS = True       # also print how bright the light is and its red/green/blue values


# =====================================================================
#  The program (you do not need to change anything below)
# =====================================================================
COLOR_NAMES = {
    color.BLACK: "BLACK", color.MAGENTA: "MAGENTA", color.PURPLE: "PURPLE",
    color.BLUE: "BLUE", color.AZURE: "AZURE", color.TURQUOISE: "TURQUOISE",
    color.GREEN: "GREEN", color.YELLOW: "YELLOW", color.ORANGE: "ORANGE",
    color.RED: "RED", color.WHITE: "WHITE", color.UNKNOWN: "NONE",
}
# One letter for the light matrix (K for blacK, so it differs from Blue)
LETTERS = {
    color.BLACK: "K", color.MAGENTA: "M", color.PURPLE: "P", color.BLUE: "B",
    color.AZURE: "A", color.TURQUOISE: "T", color.GREEN: "G", color.YELLOW: "Y",
    color.ORANGE: "O", color.RED: "R", color.WHITE: "W", color.UNKNOWN: "-",
}


def color_name(c):
    return COLOR_NAMES.get(c, str(c))


def is_brick(c):
    return c != color.UNKNOWN and c not in IGNORE_COLORS


def report(c):
    """Print the new color and show it on the hub."""
    if c == color.UNKNOWN:
        line = "  NONE       (a gap - good, gaps should read as no color)"
    elif c in IGNORE_COLORS:
        line = "  " + color_name(c) + "   (ignored - in IGNORE_COLORS)"
    else:
        line = "  " + "{:<10}".format(color_name(c))
    if SHOW_DETAILS:
        r, g, b, i = color_sensor.rgbi(COLOR_SENSOR_PORT)
        line += ("   light " + str(color_sensor.reflection(COLOR_SENSOR_PORT)) + "%"
                 + "  r/g/b " + str(r) + "/" + str(g) + "/" + str(b))
    print(line)
    light_matrix.write(LETTERS.get(c, "?"))
    if c == color.UNKNOWN:
        light.color(light.POWER, color.BLACK)   # black = light off
    else:
        light.color(light.POWER, c)


def check_sweep(bricks):
    """After a motor sweep from one end to the other, check the list of bricks."""
    middle = bricks[1:-1]
    print("Full sweep, end to end: " + " ".join(color_name(c) for c in bricks))
    problems = 0
    for c in middle:
        times = middle.count(c)
        if times >= 3:
            print("  PROBLEM: " + color_name(c) + " shows up between many bricks. That is probably"
                  " the table showing through the gaps. Add it to IGNORE_COLORS (here AND in"
                  " rl_color_search.py), or move the sensor closer to the bricks.")
            problems += 1
            break
        if times > 1 or c == BOUNDARY_COLOR:
            print("  PROBLEM: " + color_name(c) + " appears more than once. Each brick needs its"
                  " own color (or the sensor misread a brick).")
            problems += 1
            break
    if len(middle) == 0:
        print("  PROBLEM: no bricks found between the two ends.")
        problems += 1
    if problems == 0:
        print("  Looks good: " + str(len(middle)) + " different bricks between the two "
              + color_name(BOUNDARY_COLOR) + " ends. Sweep it again to make sure it's the same"
              " every time!")
    print()


async def main():
    print()
    print("=================================================")
    print("  READ COLOR - check the color sensor and track")
    print("=================================================")
    print("Turn the arm by hand, or HOLD the LEFT / RIGHT button to sweep it.")
    print("Each new color the sensor settles on is printed below.")
    print("Check: holding LEFT should move the arm to the LEFT.")
    print()
    print(">> Turn the arm by hand, or hold LEFT or RIGHT to sweep it across the bricks.")
    motor.stop(ARM_MOTOR_PORT, stop=motor.COAST)

    steady = None        # the last color the sensor settled on
    candidate = None     # the color it is reading right now
    count = 0            # how many readings in a row have matched
    moving = 0           # -1 = motor sweeping left, 1 = right, 0 = stopped
    last_sweep = 0       # direction of the last motor sweep
    bricks = []          # bricks in the order they were seen
    blocked = 0          # direction we can't go: the arm is already at that end

    while True:
        # ---- Buttons drive the arm ----
        if button.pressed(button.LEFT):
            want = -1
        elif button.pressed(button.RIGHT):
            want = 1
        else:
            want = 0
        if want == blocked:
            want = 0   # already at this end of the track
        if want != moving:
            if want == 0:
                motor.stop(ARM_MOTOR_PORT, stop=motor.COAST)
            else:
                if want != last_sweep:
                    # a sweep in a new direction starts a new list of bricks
                    if want == -1:
                        print("--- sweeping LEFT ---")
                    else:
                        print("--- sweeping RIGHT ---")
                    bricks = []
                    if steady is not None and is_brick(steady):
                        bricks.append(steady)
                    last_sweep = want
                motor.run(ARM_MOTOR_PORT, want * MOTOR_DIRECTION * SWEEP_SPEED)
            moving = want

        # ---- Read the sensor ----
        c = color_sensor.color(COLOR_SENSOR_PORT)
        if c == candidate:
            count += 1
        else:
            candidate = c
            count = 1
        if count == CONFIRM_READINGS and c != steady:
            steady = c
            report(c)
            if is_brick(c) and c != BOUNDARY_COLOR:
                blocked = 0   # back on the track: both directions are OK again
            if is_brick(c) and (len(bricks) == 0 or bricks[-1] != c):
                bricks.append(c)
                print("  Bricks in order: " + " ".join(color_name(b) for b in bricks))
                if moving != 0 and c == BOUNDARY_COLOR and len(bricks) > 1:
                    # reached an end: stop so the arm doesn't run off the track
                    motor.stop(ARM_MOTOR_PORT, stop=motor.COAST)
                    blocked = moving
                    moving = 0
                    if bricks[0] == BOUNDARY_COLOR:
                        check_sweep(bricks)
                    print(">> Reached an end of the track. Hold the other button to sweep back.")

        await runloop.sleep_ms(READING_GAP_MS)


runloop.run(main())
