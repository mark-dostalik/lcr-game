import dataclasses
import itertools
from typing import Literal, Sequence

import numpy as np
from scipy import sparse, stats

Action = tuple[Literal['keep', 'center', 'right', 'left']]


@dataclasses.dataclass(frozen=True)
class Probabilities:
    """Individual probabilities of the action directions 'left', 'center', 'right', and 'keep'."""
    left: float = 1 / 6
    center: float = 1 / 6
    right: float = 1 / 6
    keep: float = 1 / 2

    def __post_init__(self):
        if not np.allclose(self.keep + self.left + self.right + self.center, 1):
            raise ValueError('Probabilities must sum to 1.')


@dataclasses.dataclass(frozen=True)
class State:
    """Game state that consists of the chip distribution among players and of the currently active player."""
    distribution: tuple[int, ...]
    player: int


class LCRBase:
    """Base class for the LCR Markov chain solver and game simulator."""

    def __init__(self, num_players: int, num_chips: int, probas: Probabilities):
        """
        Initializes the instance with the number of players, number of chips, and the action probabilities.

        Args:
            num_players: Number of players.
            num_chips: Initial number of chips per player.
            probas: Probabilities of the action directions 'left', 'center', 'right', and 'keep'.
        """
        self.num_players = num_players
        self.pot = num_players
        self.num_chips = num_chips
        self.probas = dataclasses.asdict(probas)

    def _winner(self, distribution: Sequence[int]) -> int | None:
        """Returns the winner of the given chip distribution, or None if there is no winner."""
        try:
            (winner,) = set(distribution) - {self.pot}
        except ValueError:
            winner = None
        return winner

    def _acceptor(self, active_player: int, direction: Literal['left', 'right', 'center']) -> int:
        """Returns the player (or pot) that accepts chips from the active player based on the action direction."""
        match direction:
            case 'left':
                return (active_player + 1) % self.num_players
            case 'right':
                return (active_player - 1) % self.num_players
            case 'center':
                return self.pot
            case _:
                raise NotImplementedError(f'Unknown direction: {direction}')

    def _next_player(self, active_player: int, distribution: Sequence[int]) -> int:
        """Returns the player that plays after the currently active player given a chip distribution."""
        index = 1
        while True:
            next_player = (active_player + index) % self.num_players
            if distribution.count(next_player) > 0:
                return next_player
            index += 1


class LCRMarkovChainSolver(LCRBase):
    """Representation of the LCR game as an absorbing Markov chain."""

    def __init__(self, num_players: int, num_chips: int, probas: Probabilities):
        super().__init__(num_players, num_chips, probas)
        self._initial_state = State(tuple(sorted(num_chips * tuple(range(num_players)))), 0)
        self._transient_states: dict[State, int] = {}
        self._transition_probas: dict[int, dict[Action, float]] = {}
        self._transition_matrix: sparse._csr.csr_matrix

    def _generate_transient_states(self):
        """Generates all possible non-winning (transient) game states."""
        chips_in_total = self.num_chips * self.num_players
        index = 0
        for distribution in itertools.combinations_with_replacement(range(self.num_players + 1), chips_in_total):
            if distribution != chips_in_total * (self.pot,) and self._winner(distribution) is None:
                for player in range(self.num_players):
                    if player in distribution:
                        self._transient_states[State(distribution, player)] = index
                        index += 1

    def _generate_transition_probas(self):
        """Generates possible transition probabilities between states for all possible dice rolls."""
        for num_rolls in range(1, self.num_chips + 1):
            actions = list(itertools.combinations_with_replacement(self.probas, num_rolls))
            multinomial_distribution = stats.multinomial(num_rolls, list(self.probas.values()))
            transition_probas = [
                multinomial_distribution.pmf([action.count(item) for item in self.probas]) for action in actions
            ]
            self._transition_probas[num_rolls] = dict(zip(actions, transition_probas))

    def _next_state_index(self, state: State, action: Action) -> int:
        """Determines the index of the next state given dice roll (action) and the current game state."""
        distribution = list(state.distribution)
        for direction in action:
            if direction != 'keep':
                distribution.remove(state.player)
                distribution.append(self._acceptor(state.player, direction))

        if (winner := self._winner(distribution)) is not None:
            return len(self._transient_states) + winner

        distribution.sort()
        next_state = State(tuple(distribution), self._next_player(state.player, distribution))
        return self._transient_states[next_state]

    def _generate_transition_matrix(self):
        """Generates Markov chain transition matrix between the individual transient and absorbing game states."""
        self._generate_transition_probas()
        self._generate_transient_states()

        row_indices: list[int] = []
        column_indices: list[int] = []
        probabilities: list[float] = []

        for state, row_index in self._transient_states.items():
            num_rolls = min(self.num_chips, state.distribution.count(state.player))

            for action, probability in self._transition_probas[num_rolls].items():
                row_indices.append(row_index)
                column_indices.append(self._next_state_index(state, action))
                probabilities.append(probability)

        row_indices += list(range(len(self._transient_states), len(self._transient_states) + self.num_players))
        column_indices += list(range(len(self._transient_states), len(self._transient_states) + self.num_players))
        probabilities += self.num_players * [1.]

        self._transition_matrix = sparse.csr_matrix(
            (probabilities, (row_indices, column_indices)),
            shape=(len(self._transient_states) + self.num_players, len(self._transient_states) + self.num_players)
        )

    def solve(self, method: Literal['direct', 'iterative']) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solves the Markov chain to obtain chances of winning for each player and statistics on the number of turns.

        Args:
            method: Method to be used in solving the system of linear equations. Either 'direct' or 'iterative'.

        Returns:
            Tuple of absorbing probabilities, the expected number of turns and the variance on the number of turns.
        """
        self._generate_transition_matrix()

        initial_probability = np.zeros(len(self._transient_states))
        initial_probability[self._transient_states[self._initial_state]] = 1

        Q = self._transition_matrix[:len(self._transient_states), :len(self._transient_states)]
        R = self._transition_matrix[:len(self._transient_states), len(self._transient_states):]
        I = sparse.diags(np.ones(len(self._transient_states)), format='csr')

        match method:
            case 'direct':
                xi = sparse.linalg.spsolve(I - Q.T, initial_probability)
                tau = sparse.linalg.spsolve(I - Q, np.ones(len(self._transient_states)))
                sigma = sparse.linalg.spsolve(I - Q, 2 * tau)
            case 'iterative':
                xi, _ = sparse.linalg.lgmres(I - Q.T, initial_probability, rtol=1e-8, atol=1e-8)
                tau, _ = sparse.linalg.lgmres(I - Q, np.ones(len(self._transient_states)), rtol=1e-8, atol=1e-8)
                sigma, _ = sparse.linalg.lgmres(I - Q, 2 * tau, rtol=1e-8, atol=1e-8)
            case _:
                raise NotImplementedError('Unknown solver method.')

        absorbing_probas = xi @ R
        expected_num_turns = tau[self._transient_states[self._initial_state]]
        variance_on_num_turns = (sigma - tau - tau ** 2)[self._transient_states[self._initial_state]]

        return absorbing_probas, expected_num_turns, variance_on_num_turns


class LCRSimulator(LCRBase):
    """Simulator of the LCR game."""

    def __init__(self, num_players: int, num_chips: int, probas: Probabilities):
        super().__init__(num_players, num_chips, probas)

    def simulate(self, num_games: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates the LCR game to obtain the winning proportions for each player and statistics on the number of turns.

        Args:
            num_games: Number of games to be simulated.

        Returns:
            Tuple of winning proportions, mean of the number of turns, and standard deviation of the number of turns.
        """
        wins = np.zeros(self.num_players)
        num_turns = np.zeros(num_games)

        for index in range(num_games):
            distribution = self.num_chips * list(range(self.num_players))
            player = 0
            num_rolls = 3
            turn = 1

            while True:
                direction = np.random.choice(list(self.probas.keys()), 1, p=list(self.probas.values()))[0]

                if direction != 'keep':
                    distribution.remove(player)
                    distribution.append(self._acceptor(player, direction))  # type: ignore

                if (winner := self._winner(distribution)) is not None:
                    wins[winner] += 1
                    num_turns[index] = turn
                    break

                num_rolls -= 1
                if num_rolls == 0:
                    turn += 1
                    player = self._next_player(player, distribution)
                    num_rolls = min(self.num_chips, distribution.count(player))

        return wins / np.sum(wins), np.mean(num_turns), np.std(num_turns)


def main():
    num_players = 4
    num_chips = 3
    probas = Probabilities()

    lcr_solver = LCRMarkovChainSolver(num_players=num_players, num_chips=num_chips, probas=probas)
    absorbing_probas, expected_num_turns, variance_on_num_turns = lcr_solver.solve(method='iterative')

    print(f"Players' probabilities of winning: {np.array2string(absorbing_probas, precision=5, floatmode='fixed')}")
    print(f"Expected number of turns: {expected_num_turns:.3f}")
    print(f"Standard deviation of the number of turns: {np.sqrt(variance_on_num_turns):.3f}")

    print("--------------------------------------------------------------------------------")

    lcr_simulation = LCRSimulator(num_players=num_players, num_chips=num_chips, probas=probas)
    winning_proportions, mean_num_turns, std_num_turns = lcr_simulation.simulate(num_games=100_000)

    print(f"Players' winning proportions: {np.array2string(winning_proportions, precision=5, floatmode='fixed')}")
    print(f"Sample mean of the number of turns: {mean_num_turns:.3f}")
    print(f"Sample standard deviation of the number of turns: {std_num_turns:.3f}")


if __name__ == '__main__':
    main()
