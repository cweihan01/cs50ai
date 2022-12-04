from logic import *

# Initialise symbols
AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # If AKnight (truth), A is both knight and knave
    Implication(AKnight, And(AKnight, AKnave)),
    # If AKnave (lie), A is not both knight and knave
    Implication(AKnave, Not(And(AKnight, AKnave))),
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # If AKnight (truth), both A and B are knaves
    Implication(AKnight, And(AKnave, BKnave)),
    # If AKnave (lie), A and B are not both knaves
    Implication(AKnave, Not(And(AKnave, BKnave))),
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # If AKnight (truth), both A and B are same
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    Implication(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))),
    # If BKnight (truth), A and B are different
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    Implication(BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight)))),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # If BKnight (truth), A says he is a knave
    Implication(BKnight, Implication(AKnight, AKnave)),
    Implication(BKnight, Implication(AKnave, AKnight)),
    Implication(BKnave, Not(Implication(AKnight, AKnave))),
    Implication(BKnight, Not(Implication(AKnave, AKnight))),
    # If BKnight (truth), C is a knave
    Implication(BKnight, CKnave),
    Implication(BKnave, Not(CKnave)),
    # If CKnight (truth), A is a knight
    Implication(CKnight, AKnight),
    Implication(CKnave, Not(AKnight)),
)


# Store problem knowledge (structure of problem)
problem_knowledge = And()
characters = [(AKnight, AKnave), (BKnight, BKnave), (CKnight, CKnave)]
for c in characters:
    problem_knowledge.add(Or(c[0], c[1]))
    problem_knowledge.add(Not(And(c[0], c[1])))

# Add problem knowledge to each knowledge base
for knowledge in [knowledge0, knowledge1, knowledge2, knowledge3]:
    knowledge.add(problem_knowledge)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3),
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"{symbol}")


if __name__ == "__main__":
    main()
