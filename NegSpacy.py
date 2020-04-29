# -*- coding: utf-8 -*-
"""

@author: Nastaran Babanejad
"""

from spacy.tokens import Token, Doc, Span
from spacy.matcher import PhraseMatcher
import logging

from negspacy.termsets import LANGUAGES


class Negex:

    def __init__(
        self,
        nlp,
        language="en",
        ent_types=list(),
        pseudo_negations=list(),
        preceding_negations=list(),
        following_negations=list(),
        termination=list(),
        chunk_prefix=list(),
    ):
        if not language in LANGUAGES:
            raise KeyError("Language not found")
            
        termsets = LANGUAGES[language]
        if not Span.has_extension("negex"):
            Span.set_extension("negex", default=False, force=True)

        if not pseudo_negations:
            if not "pseudo_negations" in termsets:
                raise KeyError("pseudo_negations not specified for this language.")
            pseudo_negations = termsets["pseudo_negations"]

        if not preceding_negations:
            if not "preceding_negations" in termsets:
                raise KeyError("preceding_negations not specified for this language.")
            preceding_negations = termsets["preceding_negations"]

        if not following_negations:
            if not "following_negations" in termsets:
                raise KeyError("following_negations not specified for this language.")
            following_negations = termsets["following_negations"]

        if not termination:
            if not "termination" in termsets:
                raise KeyError("termination not specified for this language.")
            termination = termsets["termination"]

        #  build spaCy matcher patterns
        self.pseudo_patterns = list(nlp.tokenizer.pipe(pseudo_negations))
        self.preceding_patterns = list(nlp.tokenizer.pipe(preceding_negations))
        self.following_patterns = list(nlp.tokenizer.pipe(following_negations))
        self.termination_patterns = list(nlp.tokenizer.pipe(termination))

        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self.matcher.add("pseudo", None, *self.pseudo_patterns)
        self.matcher.add("Preceding", None, *self.preceding_patterns)
        self.matcher.add("Following", None, *self.following_patterns)
        self.matcher.add("Termination", None, *self.termination_patterns)
        self.nlp = nlp
        self.ent_types = ent_types

        self.chunk_prefix = list(nlp.tokenizer.pipe(chunk_prefix))

    def get_patterns(self):
        """
        returns phrase patterns used for various negation dictionaries
        
        Returns
        -------
        patterns: dict
            pattern_type: [patterns]
        """
        patterns = {
            "pseudo_patterns": self.pseudo_patterns,
            "preceding_patterns": self.preceding_patterns,
            "following_patterns": self.following_patterns,
            "termination_patterns": self.termination_patterns,
        }
        for pattern in patterns:
            logging.info(pattern)
        return patterns

    def process_negations(self, doc):
        """
        Find negations in doc and clean candidate negations to remove pseudo negations
        Parameters 
        ----------
        doc: object
            spaCy Doc object
        Returns
        -------
        preceding: list
            list of tuples for preceding negations
        following: list
            list of tuples for following negations
        terminating: list
            list of tuples of terminating phrases
        """
      
        # if not doc.is_nered:
        #     raise ValueError(
        #         "Negations are evaluated for Named Entities found in text. "
        #         
        preceding = list()
        following = list()
        terminating = list()

        matches = self.matcher(doc)
        pseudo = [
            (match_id, start, end)
            for match_id, start, end in matches
            if self.nlp.vocab.strings[match_id] == "pseudo"
        ]

        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "pseudo":
                continue
            pseudo_flag = False
            for p in pseudo:
                if start >= p[1] and start <= p[2]:
                    pseudo_flag = True
                    continue
            if not pseudo_flag:
                if self.nlp.vocab.strings[match_id] == "Preceding":
                    preceding.append((match_id, start, end))
                elif self.nlp.vocab.strings[match_id] == "Following":
                    following.append((match_id, start, end))
                elif self.nlp.vocab.strings[match_id] == "Termination":
                    terminating.append((match_id, start, end))
                else:
                    logging.warnings(
                        f"phrase {doc[start:end].text} not in one of the expected matcher types."
                    )
        return preceding, following, terminating

    def termination_boundaries(self, doc, terminating):
        """
        Create sub sentences based on terminations found in text.
        Parameters
        """
        sent_starts = [sent.start for sent in doc.sents]
        terminating_starts = [t[1] for t in terminating]
        starts = sent_starts + terminating_starts + [len(doc)]
        starts.sort()
        boundaries = list()
        index = 0
        for i, start in enumerate(starts):
            if not i == 0:
                boundaries.append((index, start))
            index = start
        return boundaries

    def negex(self, doc):
      
        preceding, following, terminating = self.process_negations(doc)
        boundaries = self.termination_boundaries(doc, terminating)
        for b in boundaries:
            sub_preceding = [i for i in preceding if b[0] <= i[1] < b[1]]
            sub_following = [i for i in following if b[0] <= i[1] < b[1]]

            for e in doc[b[0] : b[1]].ents:
                if self.ent_types:
                    if e.label_ not in self.ent_types:
                        continue
                if any(pre < e.start for pre in [i[1] for i in sub_preceding]):
                    e._.negex = True
                    continue
                if any(fol > e.end for fol in [i[2] for i in sub_following]):
                    e._.negex = True
                    continue
                if self.chunk_prefix:
                    if any(
                        c.text.lower() == doc[e.start].text.lower()
                        for c in self.chunk_prefix
                    ):
                        e._.negex = True
        return doc

    def __call__(self, doc):
        return self.negex(doc)
