"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


# TODO - done
def player(board):
    """
    Returns player who has the next turn on a board.
    """

    # If terminal board, no next turn
    if terminal(board):
        return None

    # Count number of X's and O's
    xcount = 0
    ocount = 0

    for i in range(3):
        xcount += board[i].count(X)
        ocount += board[i].count(O)

    # If xcount > ocount, O's turn
    if xcount > ocount:
        return O
    # If xcount < ocount or both are equal (start of game), X's turn
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    # If terminal board, no next action
    if terminal(board):
        return None

    # Create empty set
    actions = set()

    # If cell is empty, add coordinates of cell
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                action = (i, j)
                actions.add(action)

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    # If terminal board, no next action
    if terminal(board):
        return None

    # Unpack action
    (i, j) = action

    # Check that action is valid for board
    if board[i][j] != EMPTY:
        raise Exception("invalid action for board")

    # Create a copy of board
    copy = deepcopy(board)

    # Update copy of board
    copy[i][j] = player(board)

    return copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    result = win(board)
    return result[1]


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    # Check if game is won
    result = win(board)
    if result[0] == True:
        return True

    # Check if all cells filled
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                return False

    # Tie game with all cells filled
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    winner0 = winner(board)

    if winner0 == X:
        return 1
    elif winner0 == O:
        return -1
    else:
        return 0


# Helper functions for result(), terminal(), utility()
def win(board):
    """
    Checks if there is a matching row of 3, returning a tuple: (True/False, X/O/None)
    Uses helper functions horizontal(), vertical, diagonal()
    """

    # result is a tuple with values (is_game_won, who_won_game)
    result = False, None

    # Check for each possible win condition
    result = horizontal(board)
    if result[0] == False:
        result = vertical(board)
        if result[0] == False:
            result = diagonal(board)

    return result


def horizontal(board):
    if [X, X, X] in board:
        return True, X

    elif [O, O, O] in board:
        return True, O

    return False, None


def vertical(board):
    for j in range(3):

        if board[0][j] == board[1][j] == board[2][j] == X:
            return True, X

        elif board[0][j] == board[1][j] == board[2][j] == O:
            return True, O

    return False, None


def diagonal(board):
    if (
        board[0][0] == board[1][1] == board[2][2] == X
        or board[2][0] == board[1][1] == board[0][2] == X
    ):
        return True, X

    elif (
        board[0][0] == board[1][1] == board[2][2] == O
        or board[2][0] == board[1][1] == board[0][2] == O
    ):
        return True, O

    return False, None


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    # If terminal board, no next action
    if terminal(board):
        return None

    else:
        # Maximising player X
        if player(board) == X:
            _, move = max_value(board)
            return move

        # Minimising player O
        else:
            _, move = min_value(board)
            return move


def max_value(board):

    if terminal(board):
        return utility(board), None

    # Initialise variables
    v = -math.inf
    move = None

    for action in actions(board):
        # Consider minimiser's next actions
        v_tmp, _ = min_value(result(board, action))

        # Update variables if action maximises utility
        if v_tmp > v:
            v = v_tmp
            move = action

            # Maximum utility achieved, return immediately
            if v == 1:
                return v, move

    return v, move


def min_value(board):

    if terminal(board):
        return utility(board), None

    # Initialise variables
    v = math.inf
    move = None

    for action in actions(board):
        # Consider maximiser's next actions
        v_tmp, _ = max_value(result(board, action))

        # Update variables if action minimises utility
        if v_tmp < v:
            v = v_tmp
            move = action

            # Minimum utility achieved, return immediately
            if v == -1:
                return v, move

    return v, move
