import csv
from pprint import pprint
from typing import List
import argparse

import openreview

parser = argparse.ArgumentParser(description='ICLR Get papers of particular affiliation')
parser.add_argument('--username', type=str, help="User's e-mail")
parser.add_argument('--password', type=str, help="User's password")
args = parser.parse_args()

# add credentials
CLIENT = openreview.Client(baseurl='https://api.openreview.net',
                           username=args.username,
                           password=args.password)


def get_accepted_papers(
        path_in="../data/NeurIPS_2022_paper_status.csv",
        # path_in="../data/ICLR_2021_paper_status.csv",
):
    papers = []
    with open(path_in, "r", newline="\n") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if "Accept" in row["decision"]:
                if "Poster" in row["decision"]:
                    decision = "Poster"
                elif "Spotlight" in row["decision"]:
                    decision = "Spotlight"
                elif "Oral" in row["decision"]:
                    decision = "Oral"
                else:
                    decision = row["decision"]
                papers.append(dict(
                    paper_id=int(row["number"]),
                    title=row["title"],
                    decision=decision,
                    forum=row["forum"].split("id=")[1]
                ))
            else:
                decision = "Reject"
    print("-" * 10)
    print(f"len(papers) from csv: ", len(papers))
    print(f"papers[0]: ", papers[0])
    print("-" * 10)
    return papers


def get_papers_of_affiliation(papers, affiliation_keywords="kaist", year=2021):
    row_list = []
    for p in papers:
        print(p)
        note_list = CLIENT.get_notes(forum=p["forum"])
        author_ids = None
        for n in note_list:
            content = n.content
            if "authorids" in content:
                author_ids = content["authorids"]

        author_of_affiliation_list = []
        for aid in author_ids:
            try:
                profile = CLIENT.get_profile(aid)
                content = profile.content
                emails = content["emails"]
                history = content["history"]
                try:
                    current_affiliation = [h["institution"].get("domain", h["institution"]["name"].lower())
                                           for h in history
                                           if ("end" in h) and ((h["end"] is None) or (h["end"] > year))]
                except KeyError:
                    print(f"KeyError in {aid}")
                    pprint(history)
                    current_affiliation = []
                is_affiliation = any((affiliation_keywords in o) for o in current_affiliation)
                if is_affiliation:
                    author_of_affiliation_list.append(aid)
            except Exception as e:
                print(f"{e} in {aid}")

        if len(author_of_affiliation_list) >= 1:
            row = "\t".join([
                str(p["paper_id"]), p["title"], *author_of_affiliation_list
            ])
            row_list.append(row)
            print(f"Row append: {row}")

    print("\n-----------------------------------")
    print("Printing get_papers_of_affiliation")
    for i, r in enumerate(row_list):
        print(i + 1, r)


if __name__ == '__main__':
    get_papers_of_affiliation(get_accepted_papers())
