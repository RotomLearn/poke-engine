from dataclasses import dataclass
import os
import pathlib
import json
from .state import (
    State,
    Side,
    SideConditions,
    VolatileStatusDurations,
    Pokemon,
    Move
)

# noinspection PyUnresolvedReferences
from ._poke_engine import (
    gi as _gi,
    calculate_damage as _calculate_damage,
    state_from_string as _state_from_string,
    mcts as _mcts,
    mcts_az as _mcts_az,
    load_model as _load_model,
    id as _id
)


@dataclass
class IterativeDeepeningResult:
    """
    Result of an Iterative Deepening Expectiminimax Search

    :param side_one: The moves for side_one
    :type side_one: list[str]
    :param side_two: The moves for side_two
    :type side_two: list[str]
    :param matrix: A vector representing the payoff matrix of the search.
        Pruned branches are represented by None
    :type matrix: int
    :param depth_searched: The depth that was searched to
    :type depth_searched: int
    """

    side_one: list[str]
    side_two: list[str]
    matrix: list[float]
    depth_searched: int

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=rust_result.s1,
            side_two=rust_result.s2,
            matrix=rust_result.matrix,
            depth_searched=rust_result.depth_searched,
        )

    def get_safest_move(self) -> str:
        """
        Get the safest move for side_one
        The safest move is the move that minimizes the loss for the turn

        :return: The safest move
        :rtype: str
        """
        safest_value = float("-inf")
        safest_s1_index = 0
        vec_index = 0
        for i in range(len(self.side_one)):
            worst_case_this_row = float("inf")
            for _ in range(len(self.side_two)):
                score = self.matrix[vec_index]
                if score < worst_case_this_row:
                    worst_case_this_row = score

            if worst_case_this_row > safest_value:
                safest_s1_index = i
                safest_value = worst_case_this_row

        return self.side_one[safest_s1_index]


@dataclass
class MctsSideResult:
    """
    Result of a Monte Carlo Tree Search for a single side

    :param move_choice: The move that was chosen
    :type move_choice: str
    :param total_score: The total score of the chosen move
    :type total_score: float
    :param visits: The number of times the move was chosen
    :type visits: int
    """

    move_choice: str
    total_score: float
    visits: int


@dataclass
class MctsResult:
    """
    Result of a Monte Carlo Tree Search

    :param side_one: Result for side one
    :type side_one: list[MctsSideResult]
    :param side_two: Result for side two
    :type side_two: list[MctsSideResult]
    :param total_visits: Total number of monte carlo iterations
    :type total_visits: int
    """

    side_one: list[MctsSideResult]
    side_two: list[MctsSideResult]
    total_visits: int

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                )
                for i in rust_result.s1
            ],
            side_two=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                )
                for i in rust_result.s2
            ],
            total_visits=rust_result.iteration_count,
        )

@dataclass
class MctsSideResultAZ:
    """
    Result of a Monte Carlo Tree Search for a single side

    :param move_choice: The move that was chosen
    :type move_choice: str
    :param total_score: The total score of the chosen move
    :type total_score: float
    :param visits: The number of times the move was chosen
    :type visits: int
    """

    move_choice: str
    total_score: float
    visits: int
    prior_prob: float


@dataclass
class MctsResultAZ:
    """
    Result of a Monte Carlo Tree Search

    :param side_one: Result for side one
    :type side_one: list[MctsSideResult]
    :param side_two: Result for side two
    :type side_two: list[MctsSideResult]
    :param total_visits: Total number of monte carlo iterations
    :type total_visits: int
    """

    side_one: list[MctsSideResultAZ]
    side_two: list[MctsSideResultAZ]
    total_visits: int

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=[
                MctsSideResultAZ(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                    prior_prob=i.prior_prob
                )
                for i in rust_result.s1
            ],
            side_two=[
                MctsSideResultAZ(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                    prior_prob=i.prior_prob
                )
                for i in rust_result.s2
            ],
            total_visits=rust_result.iteration_count,
        )


def generate_instructions(state: State, side_one_move: str, side_two_move: str):
    """
    TODO
    """
    return _gi(state._into_rust_obj(), side_one_move, side_two_move)


def monte_carlo_tree_search(state: State, duration_ms: int = 1000) -> MctsResult:
    """
    Perform monte-carlo-tree-search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search
    :type duration_ms: int
    :return: the result of the search
    :rtype: MctsResult
    """
    return MctsResult._from_rust(_mcts(state._into_rust_obj(), duration_ms))

def load_model(model_path: str = None) -> str:
    """
    Load the neural network model for AlphaZero Monte Carlo Tree Search
    
    :param model_path: Path to the model file
    :type model_path: str
    :return: Model ID to use in subsequent MCTS calls
    :rtype: str
    """
    # If model_path is provided, use it directly
    if model_path is None:
        # Try to find a config file
        config_paths = [
            os.path.join(os.path.expanduser("~"), ".poke_engine_config.json"),
            os.path.join(os.getcwd(), "poke_engine_config.json")
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'model_path' in config:
                            model_path = config['model_path']
                            break
                except Exception as e:
                    print(f"Error reading config file {config_path}: {e}")
        
        # If still no model_path, check environment variable
        if model_path is None:
            model_path = os.environ.get("POKE_ENGINE_MODEL_PATH")
            
        # If still no model_path, raise an error
        if model_path is None:
            raise FileNotFoundError(
                "Model file path not found. Please provide it via one of these methods:\n"
                "1. Pass it directly to the function\n"
                "2. Set the POKE_ENGINE_MODEL_PATH environment variable\n"
                f"3. Create a config file at {config_paths}"
            )
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    # Load the model and return the model ID (which is just the path)
    return _load_model(model_path)

def monte_carlo_tree_search_az(state: State, duration_ms: int = 1000, model_id: str = None) -> MctsResultAZ:
    """
    Perform monte-carlo-tree-search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search
    :type duration_ms: int
    :param model_id: Model ID returned from load_model(), or path to model file
    :type model_id: str
    :return: the result of the search
    :rtype: MctsResultAZ
    """
    if model_id is None:
        # Try to find a default model path and load it
        model_path = find_default_model_path()  # Implement this function to find model
        model_id = load_model(model_path)
    elif os.path.exists(model_id) and model_id.endswith('.pt'):
        # If model_id looks like a file path, try to load it
        model_id = load_model(model_id)
        
    # Run MCTS with the loaded model
    return MctsResultAZ._from_rust(_mcts_az(state._into_rust_obj(), duration_ms, model_id))


def iterative_deepening_expectiminimax(
    state: State, duration_ms: int = 1000
) -> IterativeDeepeningResult:
    """
    Perform an iterative-deepening expectiminimax search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search
    :type duration_ms: int
    :return: the result of the search
    :rtype: IterativeDeepeningResult
    """
    return IterativeDeepeningResult._from_rust(_id(state._into_rust_obj(), duration_ms))


def calculate_damage(
    state: State, s1_move: str, s2_move: str, s1_moves_first: bool
) -> (list[int], list[int]):
    """
    Calculate the damage rolls for two moves

    :param state:
    :type state: State
    :param s1_move:
    :type s1_move: str
    :param s2_move:
    :type s2_move: str
    :param s1_moves_first:
    :type s1_moves_first: bool
    :return: (list[int], list[int]) - the damage rolls for the two moves
    """
    return _calculate_damage(state._into_rust_obj(), s1_move, s2_move, s1_moves_first)

def state_from_string(state_str: str) -> State:
    """
    Create a State object from its string representation

    :param state_str: String representation of the state
    :type state_str: str
    :return: A State object with _into_rust_obj method
    :rtype: State
    """
    rust_state = _state_from_string(state_str)
    
    def convert_move(rust_move) -> Move:
        return Move(
            id=rust_move.id,
            disabled=rust_move.disabled,
            pp=rust_move.pp
        )
    
    def convert_pokemon(rust_pokemon) -> Pokemon:
        return Pokemon(
            id=rust_pokemon.id,
            level=rust_pokemon.level,
            types=rust_pokemon.types,
            hp=rust_pokemon.hp,
            maxhp=rust_pokemon.maxhp,
            ability=rust_pokemon.ability,
            item=rust_pokemon.item,
            attack=rust_pokemon.attack,
            defense=rust_pokemon.defense,
            special_attack=rust_pokemon.special_attack,
            special_defense=rust_pokemon.special_defense,
            speed=rust_pokemon.speed,
            status=rust_pokemon.status,
            rest_turns=rust_pokemon.rest_turns,
            sleep_turns=rust_pokemon.sleep_turns,
            weight_kg=rust_pokemon.weight_kg,
            moves=[convert_move(m) for m in rust_pokemon.moves],
            terastallized=rust_pokemon.terastallized,
            tera_type=rust_pokemon.tera_type
        )
    
    def convert_side_conditions(rust_side_conditions) -> SideConditions:
        return SideConditions(
            aurora_veil=rust_side_conditions.aurora_veil,
            crafty_shield=rust_side_conditions.crafty_shield,
            healing_wish=rust_side_conditions.healing_wish,
            light_screen=rust_side_conditions.light_screen,
            lucky_chant=rust_side_conditions.lucky_chant,
            lunar_dance=rust_side_conditions.lunar_dance,
            mat_block=rust_side_conditions.mat_block,
            mist=rust_side_conditions.mist,
            protect=rust_side_conditions.protect,
            quick_guard=rust_side_conditions.quick_guard,
            reflect=rust_side_conditions.reflect,
            safeguard=rust_side_conditions.safeguard,
            spikes=rust_side_conditions.spikes,
            stealth_rock=rust_side_conditions.stealth_rock,
            sticky_web=rust_side_conditions.sticky_web,
            tailwind=rust_side_conditions.tailwind,
            toxic_count=rust_side_conditions.toxic_count,
            toxic_spikes=rust_side_conditions.toxic_spikes,
            wide_guard=rust_side_conditions.wide_guard
        )
    
    def convert_side(rust_side) -> Side:
        return Side(
            active_index=rust_side.active_index,
            baton_passing=rust_side.baton_passing,
            pokemon=[convert_pokemon(p) for p in rust_side.pokemon],
            side_conditions=convert_side_conditions(rust_side.side_conditions),
            wish=rust_side.wish,
            future_sight=rust_side.future_sight,
            force_switch=rust_side.force_switch,
            force_trapped=rust_side.force_trapped,
            volatile_statuses=rust_side.volatile_statuses,
            substitute_health=rust_side.substitute_health,
            attack_boost=rust_side.attack_boost,
            defense_boost=rust_side.defense_boost,
            special_attack_boost=rust_side.special_attack_boost,
            special_defense_boost=rust_side.special_defense_boost,
            speed_boost=rust_side.speed_boost,
            accuracy_boost=rust_side.accuracy_boost,
            evasion_boost=rust_side.evasion_boost,
            last_used_move=rust_side.last_used_move,
            switch_out_move_second_saved_move=rust_side.switch_out_move_second_saved_move
        )

    return State(
        side_one=convert_side(rust_state.side_one),
        side_two=convert_side(rust_state.side_two),
        weather=rust_state.weather,
        weather_turns_remaining=rust_state.weather_turns_remaining,
        terrain=rust_state.terrain,
        terrain_turns_remaining=rust_state.terrain_turns_remaining,
        trick_room=rust_state.trick_room,
        trick_room_turns_remaining=rust_state.trick_room_turns_remaining,
        team_preview=rust_state.team_preview
    )

__all__ = [
    "State",
    "Side",
    "SideConditions",
    "Pokemon",
    "Move",
    "MctsResult",
    "MctsSideResult",
    "MctsResultAZ",
    "MctsSideResultAZ",
    "IterativeDeepeningResult",
    "generate_instructions",
    "monte_carlo_tree_search",
    "monte_carlo_tree_search_az",
    "load_model",
    "iterative_deepening_expectiminimax",
    "calculate_damage",
    "state_from_string"
]
