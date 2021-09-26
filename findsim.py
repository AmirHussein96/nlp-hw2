#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""
# JHU NLP HW2
# Name: Amir Hussein
# Email: ahussei6@jhu.edu
# Term: Fall 2021

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from integerize import Integerizer   # look at integerize.py for more info
import numpy as np
# For type annotations, which enable you to check correctness of your code:
from typing import List, Optional
import pdb


try:
    # PyTorch is your friend. Not *using* it will make your program so slow.
    # And it's also required for this assignment. ;-)
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    #
    # We made this easier by including the environment file in this folder.
    # Install Miniconda, then create and activate the provided environment.
    import torch as th
    import torch.nn as nn
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise


log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.
# Logging is in general a good practice to check the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# - It prints to standard error (stderr), not standard output (stdout) by
#   default. This means it won't interfere with the real output of your
#   program. 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
#
# In `parse_args`, we provided two command line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'. 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--embeddings", type=Path, help="Path to word embeddings file.")
    parser.add_argument("--word", type=str, help="Word to lookup")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error("You need to provide a real file of embeddings.")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both of `plus` and `minus` or neither.")

    return args

class Lexicon:
	"""
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
	>>> my_lexicon.find_similar_words(bagpipe)
	"""

	def __init__(self, int_to_word={},word_to_int={},embeddings=None) -> None:
		"""Load information into coupled word-index mapping and embedding matrix."""
		# FINISH THIS FUNCTION
		self.word_to_int = word_to_int
		self.int_to_word = int_to_word
		self.embeddings = embeddings
        # Store your stuff! Both the word-index mapping and the embedding matrix.
        #
        # Do something with this size info?
        # PyTorch's th.Tensor objects rely on fixed-size arrays in memory.
        # One of the worst things you can do for efficiency is
        # append row-by-row, like you would with a Python list.
        #
        # Probably make the entire list all at once, then convert to a th.Tensor.
        # Otherwise, make the th.Tensor and overwrite its contents row-by-row.

	@classmethod
	def from_file(cls, file: Path) -> Lexicon:
        # FINISH THIS FUNCTION
		lines = []
		count = 0
		word_to_int = {}
		int_to_word={}
		
		with open(file) as f:
			first_line = next(f)  # Peel off the special first line.
			for line in f:  # All of the other lines are regular.
                  # `pass` is a placeholder. Replace with real code!
				line = line.split('\t')
				lines.append(np.array(line[1:],dtype=np.float64))
				int_to_word[count]=line[0]
				word_to_int[line[0]]=count
				count+=1
		embeddings = th.tensor(lines, dtype=th.float64)
		lexicon = Lexicon(int_to_word,word_to_int, embeddings)  # Maybe put args here. Maybe follow Builder pattern
		return lexicon

	def find_similar_words(
        self, word: str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ) -> List[str]:
		"""Find most similar words, in terms of embeddings, to a query."""
        # FINISH THIS FUNCTION
		# first check if the word is in the lexicon
		
		if word in self.word_to_int:
        # The star above forces you to use `plus` and `minus` as
        # named arguments. This helps avoid mixups or readability
        # problems where you forget which comes first.
        #
        # We've also given `plus` and `minus` the type annotation
        # Optional[str]. This means that the argument may be None, or
        # it may be a string. If you don't provide these, it'll automatically
        # use the default value we provided: None.
			if (minus is None) != (plus is None):  # != is the XOR operation!
				raise TypeError("Must include both of `plus` and `minus` or neither.")
			# Keep going!
			# Be sure that you use fast, batched computations 
			# instead of looping over the rows. If you use a loop or a comprehension
			# in this function, you've probably made a mistake.
			word_ind = self.word_to_int[word]
			#pdb.set_trace()
			word_vec = th.unsqueeze(self.embeddings[word_ind,:],1)
			plus_vec = th.zeros(word_vec.shape, dtype=th.float64)
			minus_vec = th.zeros(word_vec.shape, dtype=th.float64)
			ind_to_remove = [word_ind]
			if (minus != None) and (plus != None):
				plus_ind = self.word_to_int[plus]
				plus_vec = th.unsqueeze(self.embeddings[plus_ind,:],1)
				minus_ind = self.word_to_int[minus]
				minus_vec = th.unsqueeze(self.embeddings[minus_ind,:],1) 
				ind_to_remove.append(minus_ind)
				ind_to_remove.append(plus_ind)
			#similarities = th.mm(self.embeddings, word_vec)
			final_vec = word_vec + plus_vec - minus_vec
			cos = th.nn.CosineSimilarity(dim=1, eps=1e-6)
			similarities = cos(self.embeddings, final_vec.t())
			v, ind = th.topk(th.squeeze(similarities), 13)
			ind = ind.tolist()
			for i in ind_to_remove:
				if i in ind:
					 ind.remove(i)
			top_words = [self.int_to_word[i] for i in ind[:10]]
			
			return top_words
		else:
			pass


def format_for_printing(word_list: List[str]) -> str:
    # We don't print out the list as-is; the handout
    # asks that you display it in a particular way.
    # FINISH THIS FUNCTION
    return " ".join(word_list)


def main():
	args = parse_args()
	logging.basicConfig(level=args.verbose)
	lexicon = Lexicon.from_file(args.embeddings)
	similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus)
	print(format_for_printing(similar_words))


if __name__ == "__main__":
    main()
