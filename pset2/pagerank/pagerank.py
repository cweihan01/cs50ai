import os
import random
import re
import sys
import numpy
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # Initialise probability distribution dict with all values `0`
    distribution = dict.fromkeys(list(corpus.keys()), 0)

    # Access set of links linked to by `page`
    links = corpus[page]

    # With probability `damping_factor`, choose a link at random
    # linked to by `page`.
    if links:
        link_count = len(links)
        for link in links:
            distribution[link] = damping_factor / link_count

    # With probability `1 - damping_factor`, choose a link at random
    # chosen from all pages in the corpus.
    page_count = len(corpus)
    for page in corpus:
        distribution[page] += (1 - damping_factor) / page_count

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialise pageranks dict with all values `0`
    pageranks = dict.fromkeys(list(corpus.keys()), 0)

    # Initialise samples dict to store sample count for each page
    # (no. of times each page is sampled)
    samples = dict.fromkeys(list(corpus.keys()), 0)

    # Start with a random page
    page = random.choice(list(corpus.keys()))

    # Sample n times
    for i in range(n):

        # Increment `page` count in `samples`
        samples[page] += 1

        # Transition model for given page
        model = transition_model(corpus, page, damping_factor)

        # Access model keys (pages) and values (probabilities) as lists
        page_list = list(model.keys())
        page_probabilities = list(model.values())

        # Pick a random page based on probabilities
        page = numpy.random.choice(page_list, p=page_probabilities)

    # Calculate `pageranks` values based on `samples`
    for page in pageranks:
        pageranks[page] = samples[page] / n

    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialise `pageranks` and `pagerank_changes` dict
    page_list = list(corpus.keys())
    page_count = len(corpus)
    pageranks = dict.fromkeys(page_list, 1 / page_count)
    pagerank_changes = dict.fromkeys(page_list, math.inf)

    while True:

        # Break when convergence is achieved (for all pageranks)
        if all(
            pagerank_change <= 0.001 for pagerank_change in pagerank_changes.values()
        ):
            break

        # Iterate through each page `p` in corpus
        for page in corpus:

            # Probability that surfer chose a page at random
            random_probability = (1 - damping_factor) / page_count

            # Initialise probability that surfer followed link from `i` to `p`
            link_probability = 0

            # Sum up probabilities due to each link_page `i` that links to `p`
            for link_page, links in corpus.items():

                # If page does not link to other pages, treat it like it links to every page in corpus
                if not links:
                    links = set(page_list)

                # From `i`, we travel to each link with equal probability
                # PR(i) / NumLinks(i)
                if page in links:
                    link_probability += pageranks[link_page] / len(links)

            # Calculate new pagerank for `p` using given formula
            new_pagerank = random_probability + (damping_factor * link_probability)

            # Update dicts
            pagerank_changes[page] = abs(new_pagerank - pageranks[page])
            pageranks[page] = new_pagerank

    return pageranks


if __name__ == "__main__":
    main()
