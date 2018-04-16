import os
from pathlib import Path
import logging
def add_notified_sender(username):
    lines = []
    with open("approved_senders.txt", "r") as f:
        lines = f.read().splitlines()
    if username in lines:
        logging.info("This name is already in the file. Not adding to notified senders.")
    else:
        logging.info("Adding username {0} to the notified senders list.".format(username))
        lines.append(username)
        with open("approved_senders.txt", "a") as f:
            f.write(username + '\n')
def get_notified_senders():
    lines = []
    if not os.path.exists("approved_senders.txt"):
        logging.warning("Creating approved_senders.txt")
        Path('approved_senders.txt').touch()
        #create
    with open("approved_senders.txt", "r") as f:
        lines = f.read().splitlines()
    return lines