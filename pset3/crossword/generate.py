import sys
import copy

from crossword import *


class CrosswordCreator:
    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy() for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size, self.crossword.height * cell_size),
            "black",
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    (
                        (j + 1) * cell_size - cell_border,
                        (i + 1) * cell_size - cell_border,
                    ),
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (
                                rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10,
                            ),
                            letters[i][j],
                            fill="black",
                            font=font,
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.domains:

            # Set to contain removed words
            words_to_remove = set()

            # Access variable's length
            var_length = var.length

            # Remove words that do not match variable's length
            for word in self.domains[var]:
                if len(word) != var_length:
                    words_to_remove.add(word)

            for word in words_to_remove:
                self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False

        # Overlap will be None or a tuple (x_index, y_index)
        overlap = self.crossword.overlaps[x, y]

        # If there is an overlap, check for arc consistency
        if overlap is not None:
            x_domain = self.domains[x]
            y_domain = self.domains[y]

            x_index = overlap[0]
            y_index = overlap[1]

            words_to_remove = set()

            # Iterate through every word in x's domain. If no possible word for y, remove x's word.
            for x_word in x_domain:
                if all([x_word[x_index] != y_word[y_index] for y_word in y_domain]):
                    words_to_remove.add(x_word)
                    revised = True

            # Remove words from x's domain
            if revised:
                for word in words_to_remove:
                    self.domains[x].remove(word)

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # If `arcs` is None, start with queue of all arcs
        if arcs is None:
            queue = list()
            # `overlaps` is a dict mapping (v1, v2) to (v1_index, v2_index) or None
            overlaps = self.crossword.overlaps
            for (v1, v2) in overlaps:
                if overlaps[v1, v2] is not None:
                    queue.append((v1, v2))

        # If `arcs` is provided, use it as initial list
        else:
            queue = arcs

        # Enforce arc consistency for every arc in queue
        while queue:
            (v1, v2) = queue.pop(0)

            # If a revision is made, v1's domain has changed.
            # We need to reconsider arcs involving v1 and ensure arc consistency.
            if self.revise(v1, v2):

                # If v1's domain is empty, error
                if len(self.domains[v1]) == 0:
                    return False

                # Access neighbors (overlaps) with v1
                neighbors = self.crossword.neighbors(v1)

                # Add arcs with v1 to queue (except arc with v2)
                for neighbor in neighbors:
                    if neighbor == v2:
                        continue
                    queue.append((neighbor, v1))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for var in self.crossword.variables:
            if var not in assignment or assignment[var] is None:
                return False

        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Check for distinct values
        for v1 in assignment:
            for v2 in assignment:
                if v1 == v2:
                    continue
                elif assignment[v1] == assignment[v2]:
                    return False

        # Check for correct length of values
        for var in assignment:
            if var.length != len(assignment[var]):
                return False

        # Check for conflicts between neighboring variables
        for var in assignment:
            for neighbor in self.crossword.neighbors(var):

                # Only can have conflict if neighbor's value has been assigned
                if neighbor in assignment:

                    # Overlap will be a tuple (var_index, neighbor_index)
                    overlap = self.crossword.overlaps[var, neighbor]

                    # If overlapped characters do not match, inconsistent
                    if assignment[var][overlap[0]] != assignment[neighbor][overlap[1]]:
                        return False

        # Assignment is consistent
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        domain = list(self.domains[var])
        constraints = {word: 0 for word in domain}

        for neighbor in self.crossword.neighbors(var):

            # If variable already in `assignment`, no constraints
            if neighbor in assignment:
                continue

            # If word is in neighbor's domain, update constraint
            for word in domain:
                if word in self.domains[neighbor]:
                    constraints[word] += 1

        return sorted(domain, key=lambda word: constraints[word])

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        all_variables = self.crossword.variables
        assigned_variables = assignment.keys()
        unassigned_variables = set(all_variables - assigned_variables)

        # If only one remaining variable to assign, return immediately
        if len(unassigned_variables) == 1:
            return unassigned_variables.pop()

        """ Minimum remaining value heuristic """
        # Dict which maps var: number of words in domain
        values_remaining = {var: len(self.domains[var]) for var in unassigned_variables}
        # Sort dict by number of remaining values in ascending order
        # Each item in `sorted_values_remaining` is a tuple (var [0], values_remaining [1])
        sorted_values_remaining = sorted(values_remaining.items(), key=lambda x: x[1])

        # If there is only one minimum value, return that variable
        if sorted_values_remaining[0][1] != sorted_values_remaining[1][1]:
            return sorted_values_remaining[0][0]

        """ Degree heuristic """
        # Dict which maps var: number of degrees (number of neighbors)
        degrees = {
            var: len(self.crossword.neighbors(var)) for var in unassigned_variables
        }
        # Sort dict by number of degrees in descending order
        # Each item in `sorted_degrees` is a tuple (var [0], degrees [1])
        sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        # Return the highest degree variable regardless of whether tied
        return sorted_degrees[0][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # If `assignment` complete, return
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for word in self.order_domain_values(var, assignment):
            # Add word to copy of assignment
            assignment_copy = copy.deepcopy(assignment)
            assignment_copy[var] = word

            # If new assignment is consistent, add to `assignment` and try to assign new variable (recursively)
            if self.consistent(assignment_copy):
                assignment[var] = word
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                assignment.pop(var, None)

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
