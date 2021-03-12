from collections import defaultdict
from pprint import pprint
from typing import List
import csv
from concurrent.futures import ThreadPoolExecutor
import argparse
import os

import numpy as np
import openreview
from tqdm import tqdm

from utils import try_except

parser = argparse.ArgumentParser(description='ICLR AC Statistics')
parser.add_argument('--username', type=str, help="User's e-mail")
parser.add_argument('--password', type=str, help="User's password")
args = parser.parse_args()

# add credentials
CLIENT = openreview.Client(baseurl='https://api.openreview.net',
                           username=args.username,
                           password=args.password)

submissions = list(openreview.tools.iterget_notes(CLIENT, invitation='ICLR.cc/2021/Conference/-/Blind_Submission'))
submissions += list(openreview.tools.iterget_notes(CLIENT, invitation='ICLR.cc/2021/Conference/-/Withdrawn_Submission'))
pbar = tqdm(total=len(submissions), desc='Retrieving ratings review status...')
quality_dict = {'N/A': -1, 'Poor - not very helpful': 0, 'Good': 1, 'Outstanding': 2}

TDDATE = 1605484800000  # 2020-11-16 00:00:00


def get_position(c: openreview.Note) -> str:
    if "Area_Chair" in c.signatures[0]:
        return "Area_Chair"
    elif "AnonReviewer" in c.signatures[0]:
        return "AnonReviewer"
    elif "Authors" in c.signatures[0]:
        return "Authors"
    elif "Program_Chairs" in c.signatures[0]:
        return "Program_Chairs"
    else:
        return "Others"


def get_author_id(note):
    if not hasattr(note, "tauthor"):
        return "my_work"
    try:
        ac_email = note.tauthor
        ac_profile = CLIENT.search_profiles(emails=[note.tauthor])
        return ac_profile[ac_email].id
    except:
        return note.tauthor


@try_except
def get_status(s):
    pbar.update(1)
    # reviews = client.get_notes(forum=s.id, invitation=f'ICLR.cc/2021/Conference/Paper{s.number}/-/Official_Review')
    # rratings = client.get_notes(forum=s.id, invitation=f'ICLR.cc/2021/Conference/Paper{s.number}/.*/-/Review_Rating')

    withdraws: List[openreview.Note] = CLIENT.get_notes(
        forum=s.id, invitation=f'ICLR.cc/2021/Conference/Paper{s.number}/-/Withdraw',
    )
    if len(withdraws) > 0:
        withdraw = True
    else:
        withdraw = False

    meta_reviews: List[openreview.Note] = CLIENT.get_notes(
        forum=s.id, invitation=f'ICLR.cc/2021/Conference/Paper{s.number}/-/Meta_Review',
    )
    if len(meta_reviews) != 0:
        mr = meta_reviews[0]
        ac_id = get_author_id(mr)
        meta_review_length = len(mr.content["metareview"])
    else:
        ac_id = None
        meta_review_length = np.nan

    comments: List[openreview.Note] = CLIENT.get_notes(
        forum=s.id, invitation=f'ICLR.cc/2021/Conference/Paper{s.number}/-/Official_Comment',
    )

    position_to_comment_length_list = defaultdict(list)
    for c in reversed(comments):
        pos = get_position(c)
        if ac_id is None and pos == "Area_Chair":
            ac_id = get_author_id(c)
        comment_length = len(c.content["comment"])
        position_to_comment_length_list[pos].append(comment_length)

    return s.number, s.id, withdraw, ac_id, meta_review_length, position_to_comment_length_list


def get_stats(comment_length_list: List[int]):
    if len(comment_length_list) == 0:
        return [0, np.nan, np.nan, np.nan, np.nan, np.nan]
    return [
        len(comment_length_list),
        float(np.mean(comment_length_list)),
        float(np.std(comment_length_list)),
        float(np.min(comment_length_list)),
        float(np.max(comment_length_list)),
        float(np.median(comment_length_list)),
    ]


if __name__ == '__main__':

    futures = []
    with ThreadPoolExecutor() as executor:
        for i, s in enumerate(submissions):
            futures.append(executor.submit(get_status, s))
    pbar.close()

    os.makedirs("../data", exist_ok=True)
    with open("../data/ac_stats.csv", "w", newline="\n") as f:
        errors = []
        writer = csv.writer(f, delimiter=",")
        writer.writerow([
            "paper_id", "paper_url", "withdraw", "ac_id", "meta_review_length",
            "#AC", "mean_AC", "std_AC", "min_AC", "max_AC", "median_AC",
            "#Reviewer", "mean_Reviewer", "std_Reviewer", "min_Reviewer", "max_Reviewer", "median_Reviewer",
            "#Authors", "mean_Authors", "std_Authors", "min_Authors", "max_Authors", "median_Authors",
            "#PC", "mean_PC", "std_PC", "min_PC", "max_PC", "median_PC",
            "#ETC", "mean_ETC", "std_ETC", "min_ETC", "max_ETC", "median_ETC",
        ])
        for future in futures:
            ret = future.result()
            if ret[0] is not None:
                paper_no, paper_url, wd, ac_id, mlr, pos_to_cll = future.result()
                pprint(pos_to_cll)
                row = [
                    paper_no,
                    paper_url,
                    "withdraw" if wd else "not_withdraw",
                    ac_id,
                    mlr,
                    *get_stats(pos_to_cll["Area_Chair"]),
                    *get_stats(pos_to_cll["AnonReviewer"]),
                    *get_stats(pos_to_cll["Authors"]),
                    *get_stats(pos_to_cll["Program_Chairs"]),
                    *get_stats(pos_to_cll["Others"]),
                ]
                print(row)
                print("-----")
                writer.writerow(row)
            else:
                errors.append(ret)
    print("Errors -- ")
    pprint(errors)



