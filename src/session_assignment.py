from collections import defaultdict, Counter
from copy import deepcopy
from itertools import chain
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

    @property
    def time(self):
        return {"Oral": 15, "Spotlight": 10, "Poster": 5}[self.decision]

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
    override_time: int = None

    @property
    def is_op_awarded(self):
        return self.paper.is_op_awarded

    @property
    def time(self):
        return self.override_time or self.paper.time

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


def get_paper_session(exclude_op_papers,
                      qna_time_for_oral_session,
                      path="../data/ICLR-2021-sessions.csv",
                      id_to_paper=None):
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
               max_time_per_session=3 * 60 - qna_time_for_oral_session,
               consider_cluster_first=True,
               exclude_op_papers=exclude_op_papers).dump(
        [p for _, p in id_to_paper.items() if p.decision != "Poster"],
        assign_false_paper=False,  # we will assign false paper manually for oral session.
    )

    for p in posters:
        p.override_time = 5
    Assignment(posters,
               path_out="../data/ICLR-2021-assignments-poster-{}.csv".format("w-op"),
               max_time_per_session=2 * 60,
               consider_cluster_first=True,
               exclude_op_papers=False).dump(
        [p for _, p in id_to_paper.items()],
        assign_false_paper=True,
    )


class Assignment:

    def __init__(self,
                 reservations: List[Reservation],
                 path_out: str,
                 max_time_per_session: int,
                 consider_cluster_first: bool,
                 exclude_op_papers=False):
        self.reservations = reservations
        self.id_to_reservation: Dict[int, Reservation] = {r.paper_id: r for r in self.reservations}

        self.path_out = path_out
        self.max_time_per_session = max_time_per_session
        self.consider_cluster_first = consider_cluster_first
        self.exclude_op_papers = exclude_op_papers

        self.session_and_order_to_reservation = defaultdict(lambda: defaultdict(list))
        for r in self.reservations:
            for i, s in enumerate(r.order):
                self.session_and_order_to_reservation[s][i].append(r)

        self.session_to_r_list: Dict[str, List[Reservation]] = self.assign()

    def assign(self, c_level=0) -> Dict[str, List[Reservation]]:
        session_to_r_list: Dict[str, List[Reservation]] = defaultdict(list)

        for r in self.reservations:

            is_assigned = False

            if self.consider_cluster_first:
                order_generator = ((i, g) for i, gen in chain(enumerate([r.order, r.order])) for g in gen)
            else:
                order_generator = enumerate(r.order)

            for i, o in order_generator:
                _consider_cluster_here = self.consider_cluster_first and (i == 0)
                is_assigned, session_to_r_list = self._assign(
                    r, o, session_to_r_list, c_level, consider_cluster=_consider_cluster_here)
                if is_assigned:
                    break

            if not is_assigned:
                session_to_r_list = self._reallocate(r, session_to_r_list)

        print("-" * 10)
        for s, r_list in session_to_r_list.items():
            print(s, len(r_list), self._time(s, session_to_r_list))
        print("-" * 10)

        return session_to_r_list

    def _assign(self, r, o, session_to_r_list, c_level, consider_cluster):
        _time_of_o = self._time(o, session_to_r_list)
        _clusters_of_o = self._clusters(o, session_to_r_list, c_level)
        is_time_good = _time_of_o + r.time <= self.max_time_per_session
        is_cluster_good = r.clusters[c_level] in _clusters_of_o
        is_assigned = is_time_good and (not consider_cluster or is_cluster_good)
        if is_assigned:
            session_to_r_list[o].append(r)
        return is_assigned, session_to_r_list

    def _time(self, session, session_to_r_list):
        return sum([_r.time for _r in session_to_r_list[session]])

    def _clusters(self, session, session_to_r_list, c=0):
        counter = Counter([_r.clusters[c] for _r in session_to_r_list[session]])
        return list(counter.keys())

    def _reallocate(self, r_to_allocate: Reservation, session_to_r_list: Dict[str, List[Reservation]]):
        is_assigned = False
        min_triplet, min_time = None, 10000
        for o in r_to_allocate.order:
            r_list = session_to_r_list[o]
            for r_allocated in r_list:
                for o_candidate_to_move in r_allocated.order:

                    _time_of_o_candidate = self._time(o_candidate_to_move, session_to_r_list)

                    if min_time > _time_of_o_candidate:
                        min_time = _time_of_o_candidate
                        min_triplet = (o, r_allocated, o_candidate_to_move)

                    if _time_of_o_candidate + r_allocated.time <= self.max_time_per_session:
                        session_to_r_list[o_candidate_to_move].append(r_allocated)
                        session_to_r_list[o] = [r for r in session_to_r_list[o] if r.paper_id != r_allocated.paper_id]
                        session_to_r_list[o].append(r_to_allocate)
                        is_assigned = True
                        break
                if is_assigned:
                    break
            if is_assigned:
                break
        if not is_assigned:
            (o, r_allocated, o_candidate_to_move) = min_triplet
            session_to_r_list[o] = [r for r in session_to_r_list[o] if r.paper_id != r_allocated.paper_id]
            session_to_r_list[o].append(r_to_allocate)
            session_to_r_list[o_candidate_to_move].append(r_allocated)
        return session_to_r_list

    def dump(self, papers: List[Paper], assign_false_paper=True):
        id_to_session: Dict[int, str] = {r.paper_id: s
                                         for s, r_list in self.session_to_r_list.items()
                                         for r in r_list}
        if assign_false_paper:
            for paper in papers:
                if paper.paper_id not in id_to_session and (not paper.is_op_awarded):
                    _cluster = paper.clusters()[0]
                    time_smallest_session = min(self._time(s, self.session_to_r_list) for s in self.session_to_r_list)
                    smallest_session_to_clusters = {
                        session: [r.clusters[0] for r in r_list]
                        for session, r_list in self.session_to_r_list.items()
                        if self._time(session, self.session_to_r_list) == time_smallest_session
                    }
                    smallest_session = None
                    for smallest_session, clusters in smallest_session_to_clusters.items():
                        if _cluster in clusters:
                            self.session_to_r_list[smallest_session].append(Reservation(paper=paper))
                            break
                    else:
                        self.session_to_r_list[smallest_session].append(Reservation(paper=paper))
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
                    r_list = self.session_to_r_list[session]
                    for c in range(num_clusters):
                        num_friend_dict["Friend-in-Session-{}".format(c)] = round(
                            len([r for r in r_list if r.clusters[c] == paper.clusters()[c]])
                            / len(r_list),
                            2,
                        )
                except KeyError:
                    session = False
                    for c in range(num_clusters):
                        num_friend_dict["Friend-in-Session-{}".format(c)] = "N/A"
                try:
                    reservation = self.id_to_reservation[paper.paper_id].order
                    selected_rank = reservation.index(session)
                except KeyError:
                    reservation = []
                    selected_rank = "N/A"
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
    EXCLUDE_OP_PAPERS = True
    get_paper_session(exclude_op_papers=EXCLUDE_OP_PAPERS,
                      qna_time_for_oral_session=30,
                      id_to_paper=get_papers())
