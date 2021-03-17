from collections import defaultdict, Counter
from copy import deepcopy
from pprint import pprint
from typing import List, Tuple, Dict
import csv
from dataclasses import dataclass, asdict, field

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np


def build_analyzer_with_stemming(_stemmer, _analyzer):
    def _func(sentence: str):
        return [_stemmer.stem(w) for w in _analyzer(sentence)]

    return _func


def get_tfidfvectorizer_with_stemming(stemmer=None, analyzer=None):
    stemmer = stemmer or PorterStemmer()
    analyzer = analyzer or TfidfVectorizer(stop_words='english').build_analyzer()
    return TfidfVectorizer(
        stop_words='english',
        analyzer=build_analyzer_with_stemming(stemmer, analyzer)
    )


def get_document_vector(raw_documents: List[str], vectorizer=None) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = vectorizer or get_tfidfvectorizer_with_stemming()
    return vectorizer, vectorizer.fit_transform(raw_documents).toarray()


@dataclass
class Paper:
    paper_id: int
    title: str
    abstract: str
    decision: str
    cluster06: int = None
    cluster09: int = None
    cluster12: int = None

    @property
    def is_op_awarded(self):
        return self.paper_id in [3177, 2114, 3496, 1820, 1018, 2561, 2508, 437]

    def typing(self):
        self.paper_id = int(self.paper_id)
        self.cluster06 = int(self.cluster06)
        self.cluster09 = int(self.cluster09)
        self.cluster12 = int(self.cluster12)
        return self

    def text(self):
        return "{} {}".format(self.title, self.abstract)

    def __repr__(self):
        return "P(title={})".format(self.title)

    def clusters(self):
        return self.cluster06, self.cluster09, self.cluster12


@dataclass()
class Reservation:
    paper: Paper
    order: list = None

    @property
    def is_op_awarded(self):
        return self.paper.is_op_awarded

    def build_order(self, rows):
        sorted_rows = sorted(rows, key=lambda r: int(r["rank"]))
        self.order = [r["reserve_session_name"] for r in sorted_rows]
        return self

    @property
    def clusters(self):
        return self.paper.clusters()

    def len_clusters(self):
        return len(self.paper.clusters())

    @property
    def paper_id(self):
        return self.paper.paper_id


def get_important_words(km, vectorizer, num_words) -> List[List[str]]:
    centers = km.cluster_centers_
    centers_idx = np.argsort(-centers, axis=1)[:, :num_words]
    iw_array = np.asarray(vectorizer.get_feature_names())[centers_idx]
    return iw_array.tolist()


def get_papers(
        path_in="../data/ICLR_2021_paper_status.csv",
        path_out="../data/ICLR_2021_accepted_paper.csv",
        path_center_words="../data/ICLR_2021_center_words.csv",
        num_center_words=20,
        as_dict=True,
):
    try:
        with open(path_out, "r", newline="\n") as f:
            reader = csv.DictReader(f, delimiter=",")
            papers = [Paper(**row).typing() for row in reader]
    except FileNotFoundError:
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
                        raise ValueError("Error: {}".format(row["decision"]))
                else:
                    decision = "Reject"
                papers.append(Paper(
                    paper_id=int(row["number"]),
                    title=row["title"],
                    abstract=row["abstract"],
                    decision=decision,
                ))
            vectorizer, paper_vector = get_document_vector([p.text() for p in papers])  # (2595, 8409)

            km06 = KMeans(n_clusters=6, init='k-means++', max_iter=100, n_init=1, verbose=True)
            km06.fit(paper_vector)
            km09 = KMeans(n_clusters=9, init='k-means++', max_iter=100, n_init=1, verbose=True)
            km09.fit(paper_vector)
            km12 = KMeans(n_clusters=12, init='k-means++', max_iter=100, n_init=1, verbose=True)
            km12.fit(paper_vector)
            for p, l06, l09, l12 in zip(papers, km06.labels_, km09.labels_, km12.labels_):
                p.cluster06 = l06
                p.cluster09 = l09
                p.cluster12 = l12
        with open(path_out, "w", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(papers[-1]))
            writer.writeheader()
            for p in papers:
                if p.decision != "Reject":
                    writer.writerow(asdict(p))
            print("Dump: {}".format(path_out))
        with open(path_center_words, "w", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=["n_clusters", "cluster_idx"] + list(range(num_center_words)))
            writer.writeheader()
            for km in [km06, km09, km12]:
                for cluster_idx, word_list in enumerate(get_important_words(km, vectorizer, num_center_words)):
                    writer.writerow({"n_clusters": km.n_clusters, "cluster_idx": cluster_idx,
                                     **{i: w for i, w in enumerate(word_list)}})
            print("Dump: {}".format(path_center_words))
        papers = [p for p in papers if p.decision != "Reject"]

    return papers if not as_dict else {p.paper_id: p for p in papers}


def get_paper_session(exclude_op_papers, path="../data/ICLR-2021-sessions.csv", id_to_paper=None):
    id_and_type_to_rows = defaultdict(list)
    posters, spotlights, orals = [], [], []
    with open(path, "r", newline="\n") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            id_and_type_to_rows[(abs(int(row["sourceid"])), row["type"])].append(row)

    for (paper_id, paper_type), rows in id_and_type_to_rows.items():
        r = Reservation(paper=id_to_paper[paper_id]).build_order(rows)
        if paper_type == "Poster":
            posters.append(r)
        elif paper_type == "Spotlight":
            spotlights.append(r)
        elif paper_type == "Oral":
            orals.append(r)
        else:
            raise ValueError

    op_postfix = "wo-op" if exclude_op_papers else "w-op"
    Assignment([r for r in orals + spotlights if (not exclude_op_papers) or (not r.is_op_awarded)],
               path_out="../data/ICLR-2021-assignments-oral-{}.csv".format(op_postfix),
               exclude_op_papers=exclude_op_papers).dump(
        [p for _, p in id_to_paper.items() if p.decision != "Poster"]
    )
    Assignment(posters,
               path_out="../data/ICLR-2021-assignments-poster-{}.csv".format("w-op"),
               exclude_op_papers=False).dump(
        [p for _, p in id_to_paper.items()]
    )


class Assignment:

    def __init__(self,
                 reservations: List[Reservation],
                 path_out: str,
                 max_size=None,
                 exclude_op_papers=False):
        self.reservations = reservations
        self.id_to_reservation: Dict[int, Reservation] = {r.paper_id: r for r in self.reservations}

        self.path_out = path_out
        sess_counter = Counter(sum([r.order for r in reservations], []))
        paper_per_session = int(len(reservations) / len(sess_counter))
        self.max_size = max_size or paper_per_session
        self.exclude_op_papers = exclude_op_papers

        self.session_and_order_to_reservation = defaultdict(lambda: defaultdict(list))
        for r in self.reservations:
            for i, s in enumerate(r.order):
                self.session_and_order_to_reservation[s][i].append(r)

        self.session_to_r_list: Dict[str, List[Reservation]] = self.assign()

    def assign(self) -> Dict[str, List[Reservation]]:
        raise NotImplementedError

    def dump(self, papers: List[Paper]):
        id_to_session: Dict[int, str] = {r.paper_id: s
                                         for s, r_list in self.session_to_r_list.items()
                                         for r in r_list}
        num_clusters = self.reservations[0].len_clusters()
        with open(self.path_out, "w", newline="\n") as f:
            fieldnames = list(asdict(papers[0])) + ["Session", "Reservations", "Selected-Rank"] \
                         + ["Friend-in-Session-{}".format(c) for c in range(num_clusters)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            num_friend_dict = {}
            for paper in papers:
                try:
                    session = id_to_session[paper.paper_id]
                    reservation = self.id_to_reservation[paper.paper_id].order
                    selected_rank = reservation.index(session)
                    r_list = self.session_to_r_list[session]
                    for c in range(num_clusters):
                        num_friend_dict["Friend-in-Session-{}".format(c)] = round(
                            len([r for r in r_list if r.clusters[c] == paper.clusters()[c]])
                            / len(r_list),
                            2,
                        )
                except KeyError:
                    session = False
                    reservation = []
                    selected_rank = None
                    for c in range(num_clusters):
                        num_friend_dict["Friend-in-Session-{}".format(c)] = "N/A"
                if self.exclude_op_papers and paper.is_op_awarded:
                    session = "OUTSTANDING"
                writer.writerow({"Session": session, "Reservations": reservation,
                                 "Selected-Rank": selected_rank,
                                 **num_friend_dict,
                                 **asdict(paper)})
            print("Dump: {}".format(self.path_out))

    def sort_by_cluster_size(self, r_list: List[Reservation], decreasing=True):
        c0 = Counter([r.clusters[0] for r in r_list])
        c1 = Counter([r.clusters[1] for r in r_list])
        c2 = Counter([r.clusters[2] for r in r_list])

        def key_func(r: Reservation):
            return c0[r.clusters[0]], c1[r.clusters[1]], c2[r.clusters[2]]

        _sorted = sorted(r_list, key=key_func)
        if not decreasing:
            return _sorted
        else:
            return list(reversed(_sorted))


if __name__ == '__main__':
    EXCLUDE_OP_PAPERS = False
    get_paper_session(exclude_op_papers=EXCLUDE_OP_PAPERS, id_to_paper=get_papers())
