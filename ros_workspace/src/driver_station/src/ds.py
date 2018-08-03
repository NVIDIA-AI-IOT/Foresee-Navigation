#!/usr/bin/env python
from __future__ import print_function
import curses
import rospy
from std_msgs.msg import String
from rosgraph_msgs.msg import Log

ROSOUT_BUFFER = [
    "Echoing rosout_agg...",
    "",
    ""
]

MAX_LOG_LINES = 18


def rosout_callback(data):
    global ROSOUT_BUFFER
    global MAX_LOG_LINES
    for line in data.msg.split('\n'):
        ROSOUT_BUFFER.append(line)
    if len(ROSOUT_BUFFER) > MAX_LOG_LINES:
        ROSOUT_BUFFER = ROSOUT_BUFFER[-MAX_LOG_LINES:]


def main(stdscr):
    global MAX_LOG_LINES
    stdscr.nodelay(1)

    pub = rospy.Publisher('/enabled', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(30)

    enabled = False
    keylog = [0, 0]

    rospy.Subscriber("/rosout_agg", Log, rosout_callback)

    while not rospy.is_shutdown():
        c = stdscr.getch()
        if c != curses.ERR:
            keylog.append(c)
            keylog.pop(0)

        # ctrl-x ctrl-e
        if keylog == [24, 5]:
            enabled = True
            keylog = [0, 0]
        # ctrl-d
        elif keylog[-1:] == [ord(" ")]:
            enabled = False
            keylog = [0, 0]

        MAX_LOG_LINES = stdscr.getmaxyx()[0]-4
        # stdscr.clear()
        xlen = stdscr.getmaxyx()[1]
        for i in xrange(len(ROSOUT_BUFFER)):
            stdscr.addstr(i, 0, ROSOUT_BUFFER[i].ljust(xlen))
        render_status_text(stdscr, enabled)
        render_divider(stdscr)
        render_keylog(stdscr, keylog)
        stdscr.refresh()

        if enabled:
            pub.publish("1")
        rate.sleep()


def render_status_text(stdscr, is_enabled):
    y = stdscr.getmaxyx()[0] - 3
    if is_enabled:
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        stdscr.addstr(y, 0, "Enabled ", curses.color_pair(1)
                      | curses.A_STANDOUT)
    else:
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        stdscr.addstr(y, 0, "Disabled", curses.color_pair(1))


def render_divider(stdscr):
    y = stdscr.getmaxyx()[0] - 4
    x = stdscr.getmaxyx()[1]
    stdscr.addstr(y, 0, "-" * x, curses.color_pair(1))


def render_keylog(stdscr, keylog):
    y = stdscr.getmaxyx()[0] - 1
    stdscr.addstr(y, 0, str(keylog).ljust(10))


if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException as e:
        print("[Exited due to ROS Interrupt]")
    except KeyboardInterrupt as e:
        print("[KeyboardInterrupt]")
