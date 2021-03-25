import csv
from pprint import pprint
from typing import List
import argparse

import openreview

parser = argparse.ArgumentParser(description='ICLR Get emails of papers')
parser.add_argument('--username', type=str, help="User's e-mail")
parser.add_argument('--password', type=str, help="User's password")
args = parser.parse_args()

# add credentials
CLIENT = openreview.Client(baseurl='https://api.openreview.net',
                           username=args.username,
                           password=args.password)


def get_paper_of_session(session_numbers: List[int],
                         path_in="../data/ICLR-2021-assignments-oral-revised.tsv",
                         session_prefix=None):
    session_prefix = session_prefix or "Oral Session"
    sessions = ["{} {}".format(session_prefix, i) for i in session_numbers]
    with open(path_in, "r", newline="\n") as f:
        reader = csv.DictReader(f, delimiter="\t")
        papers = [row for row in reader if row["Session"] in sessions]
        return papers


def get_email_set_from_id(forum_and_paper_id_list):
    return [
        set(note.tauthor for note in CLIENT.get_notes(
            forum=forum,
            signature=f'ICLR.cc/2021/Conference/Paper{paper_id}/Authors',
        ))
        for forum, paper_id in forum_and_paper_id_list
    ]


def get_author_emails_of_papers(paper_number_list: List[int], verbose=True):
    submissions: List[openreview.Note] = [
        note for note in openreview.tools.iterget_notes(
            CLIENT, invitation='ICLR.cc/2021/Conference/-/Blind_Submission',
        ) if note.number in paper_number_list
    ]
    forum_and_paper_id_list = [(note.forum, note.number) for note in submissions]
    email_list = get_email_set_from_id(forum_and_paper_id_list)
    if verbose:
        for (f, p), email_set in zip(forum_and_paper_id_list, email_list):
            print("\t".join([str(p), *email_set]))
    return email_list


if __name__ == '__main__':

    papers_25811 = get_paper_of_session([2, 5, 8, 11])
    paper_numbers = [int(p["paper_id"]) for p in papers_25811]

    print(paper_numbers)
    print("---------")

    get_author_emails_of_papers(paper_numbers)
