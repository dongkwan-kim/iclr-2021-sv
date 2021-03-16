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


def get_paper_session(preference, exclude_op_papers, path="../data/ICLR-2021-sessions.csv", id_to_paper=None):
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
               path_out="../data/ICLR-2021-assignments-oral-{}-{}.csv".format(preference, op_postfix),
               preference=preference,
               exclude_op_papers=exclude_op_papers).dump(
        [p for _, p in id_to_paper.items() if p.decision != "Poster"]
    )
    Assignment(posters,
               path_out="../data/ICLR-2021-assignments-poster-{}-{}.csv".format(preference, "w-op"),
               preference=preference,
               exclude_op_papers=False).dump(
        [p for _, p in id_to_paper.items()]
    )


class Assignment:

    def __init__(self,
                 reservations: List[Reservation],
                 path_out: str,
                 max_size=None,
                 preference="not_small_cluster",
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

        self.preference = preference
        if preference == "large_cluster":
            self.session_to_r_list: Dict[str, List[Reservation]] = self.assign_to_prefer_large_cluster()
        elif preference == "not_small_cluster":
            self.session_to_r_list: Dict[str, List[Reservation]] = self.assign_to_prefer_not_small_cluster()
        else:
            raise ValueError

        self.id_to_session: Dict[int, str] = dict()

        for s, r_list in self.session_to_r_list.items():
            for r in r_list:
                self.id_to_session[r.paper_id] = s

    def dump(self, papers: List[Paper]):
        num_clusters = self.reservations[0].len_clusters()
        with open(self.path_out, "w", newline="\n") as f:
            fieldnames = list(asdict(papers[0])) + ["Session", "Reservations", "Selected-Rank"] \
                         + ["Friend-in-Session-{}".format(c) for c in range(num_clusters)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            num_friend_dict = {}
            for paper in papers:
                try:
                    session = self.id_to_session[paper.paper_id]
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

    def assign_reservation_wise(self, is_r_assigned, session_to_r_list):
        # Reservation-wise assignment
        for r in self.reservations:
            if not is_r_assigned[r.paper_id]:
                min_session, min_session_size = None, 100000
                min_session_wo_max_cap, min_session_wo_max_cap_size = None, 100000
                for i, o in enumerate(r.order):
                    if self.max_size > min_session_size > len(session_to_r_list[o]):
                        min_session_size = len(session_to_r_list[o])
                        min_session = o
                    if min_session_wo_max_cap_size > len(session_to_r_list[o]):
                        min_session_wo_max_cap_size = len(session_to_r_list[o])
                        min_session_wo_max_cap = o
                    if i == 2 and min_session is not None:
                        break
                if min_session is None:
                    min_session = min_session_wo_max_cap
                session_to_r_list[min_session].append(r)
                is_r_assigned[r.paper_id] = True
        return is_r_assigned, session_to_r_list

    def redistribute(self, session_to_r_list: Dict[str, List[Reservation]]):\

        large_session_queue = [session for session, r_list in session_to_r_list.items()
                               if len(r_list) > self.max_size]

        def same_session_same_cluster(r, short_session, short_session_clusters):
            return short_session in r.order and r.clusters[0] in short_session_clusters

        def same_session(r, short_session, _):
            return short_session in r.order

        def _redistribute_from_larges(cond, max_iter=30):
            i = 0
            while len(large_session_queue) > 0:
                large_session = large_session_queue.pop(0)
                for short_session, short_r_list in sorted(session_to_r_list.items(), key=lambda t: len(t[1])):
                    if len(short_r_list) < self.max_size:

                        large_session_list = session_to_r_list[large_session]
                        large_session_clusters = [r.clusters[0] for r in large_session_list]
                        nam_session_to_abandon = len(large_session_clusters) - self.max_size

                        short_session_clusters = [r.clusters[0] for r in short_r_list]
                        legible_r_list = [r for r in large_session_list
                                          if cond(r, short_session, short_session_clusters)][:1]
                        legible_r_id_set = set([r.paper_id for r in legible_r_list])
                        session_to_r_list[large_session] = [r for r in session_to_r_list[large_session]
                                                            if r.paper_id not in legible_r_id_set]
                        session_to_r_list[short_session] += legible_r_list

                        if len(session_to_r_list[large_session]) > self.max_size:
                            large_session_queue.append(large_session)

                        if nam_session_to_abandon <= 0:
                            break
                if len(session_to_r_list[large_session]) > self.max_size:
                    large_session_queue.append(large_session)
                i += 1
                if i > max_iter:
                    break

        _redistribute_from_larges(same_session_same_cluster, max_iter=30)
        _redistribute_from_larges(same_session, max_iter=1)
        return session_to_r_list

    def assign_to_prefer_not_small_cluster(self):
        session_to_r_list = defaultdict(list)

        is_r_assigned = defaultdict(lambda: False)
        cluster_to_r_list = defaultdict(list)
        for r in self.reservations:
            cluster_to_r_list[r.clusters[0]].append(r)

        for c, r_list in sorted(cluster_to_r_list.items(), key=lambda t: -len(t[1])):
            r_list: List[Reservation]
            max_num_order = max([len(r.order) for r in r_list])
            min_num_division = len(r_list) // self.max_size + 1

            session_counter_list = []
            for i_o in range(1, max_num_order):
                session_counter_list.append(Counter(sum([r.order[:i_o] for r in r_list], [])))

            is_broken = False
            num_division, most_common_sessions, total_size_of_most_common_session = None, None, None
            for num_division in range(1, len(r_list) + 1):
                for i, session_counter in enumerate(session_counter_list):
                    most_common_sessions = session_counter.most_common(num_division)
                    total_size_of_most_common_session = sum(mcs_val for mcs, mcs_val in most_common_sessions)

                    all_slots_available = [mcs_val + len(session_to_r_list[mcs]) <= self.max_size
                                           for mcs, mcs_val in most_common_sessions]

                    if total_size_of_most_common_session >= len(r_list) \
                            and num_division >= min_num_division \
                            and all(all_slots_available):
                        is_broken = True
                        break
                if is_broken:
                    break

            # print(num_division, most_common_sessions, len(r_list))
            order_depth = len(most_common_sessions)
            most_common_session_names = [mcs for mcs, _ in most_common_sessions]
            sorted_r_list = sorted([r for r in r_list],
                                   key=lambda _r: -len(set(_r.order[:order_depth + 1] + most_common_session_names)))
            for r in sorted_r_list:
                for o in r.order[:order_depth + 1]:
                    if o in most_common_session_names and len(session_to_r_list[o]) <= self.max_size:
                        session_to_r_list[o].append(r)
                        is_r_assigned[r.paper_id] = True
                        break

        is_r_assigned, session_to_r_list = self.assign_reservation_wise(is_r_assigned, session_to_r_list)
        session_to_r_list = self.redistribute(session_to_r_list)

        return session_to_r_list

    def assign_to_prefer_large_cluster(self):
        session_to_r_list = defaultdict(list)
        is_r_assigned = defaultdict(lambda: False)

        # Session-wise assignment
        for s, order_to_r_list in sorted(self.session_and_order_to_reservation.items(),
                                         key=lambda t: sum(len(r) for r in t[1].values())):
            c_r_list: List[Reservation] = []
            is_assigned = False
            for o, r_list in order_to_r_list.items():
                c_r_list += r_list
                for c in reversed(range(r_list[-1].len_clusters())):
                    clusters = [r.clusters[c] for r in c_r_list if not is_r_assigned[r.paper_id]]
                    if len(clusters) > 0:
                        max_key, max_val = Counter(clusters).most_common(1)[0]
                        if max_val >= self.max_size:
                            is_assigned = True
                            assigned_list = self.sort_by_cluster_size([r for r in c_r_list
                                                                       if r.clusters[c] == max_key
                                                                       and not is_r_assigned[r.paper_id]])
                            assigned_list = assigned_list[:self.max_size]
                            session_to_r_list[s] = assigned_list
                            for r in assigned_list:
                                is_r_assigned[r.paper_id] = True
                            break
                if is_assigned:
                    break
            else:  # not assigned
                clusters = [r.clusters[0] for r in c_r_list if not is_r_assigned[r.paper_id]]
                max_key, max_val = Counter(clusters).most_common(1)[0]
                assigned_list = self.sort_by_cluster_size([r for r in c_r_list
                                                           if r.clusters[0] == max_key
                                                           and not is_r_assigned[r.paper_id]])
                session_to_r_list[s] = assigned_list
                for r in assigned_list:
                    is_r_assigned[r.paper_id] = True

        is_r_assigned, session_to_r_list = self.assign_reservation_wise(is_r_assigned, session_to_r_list)
        session_to_r_list = self.redistribute(session_to_r_list)

        return session_to_r_list


if __name__ == '__main__':
    EXCLUDE_OP_PAPERS = False
    get_paper_session(preference="large_cluster", exclude_op_papers=EXCLUDE_OP_PAPERS, id_to_paper=get_papers())
    get_paper_session(preference="not_small_cluster", exclude_op_papers=EXCLUDE_OP_PAPERS, id_to_paper=get_papers())
