use crate::mcts::{perform_mcts, perform_mcts_evolved};
use crate::mcts_pn::{perform_mcts_pn_batched, PolicyNetwork, DEFAULT_TOP_K_MOVES};
use crate::mcts_pruned::{perform_mcts_pruned_batched, JointNetwork};
use crate::state::{MoveChoice, State};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tch::Device;

pub struct PlayerChoice {
    pub chosen_move: MoveChoice,
    pub move_weights: Vec<f32>, // Will be normalized visit counts for MCTS, uniform for random
}

// Updated trait with explicit Send + Sync requirement
pub trait Player: Send + Sync {
    fn choose_move(
        &mut self,
        state: &mut State,
        is_player_one: bool,
        turn_count: u32,
    ) -> PlayerChoice;
}

pub enum MctsType {
    Baseline, // Uses normal evaluate.rs
    Evolved,  // Uses evolved_evaluate.rs
}

pub struct MctsPlayer {
    search_time: Duration,
    mcts_type: MctsType,
}

impl MctsPlayer {
    pub fn new(search_time: Duration) -> Self {
        Self {
            search_time,
            mcts_type: MctsType::Baseline,
        }
    }

    pub fn with_type(search_time: Duration, mcts_type: MctsType) -> Self {
        Self {
            search_time,
            mcts_type,
        }
    }
}

impl Player for MctsPlayer {
    fn choose_move(&mut self, state: &mut State, is_player_one: bool, _: u32) -> PlayerChoice {
        let (side_one_options, side_two_options) = state.get_all_options();

        let result = match self.mcts_type {
            MctsType::Baseline => perform_mcts(
                state,
                side_one_options.clone(),
                side_two_options.clone(),
                self.search_time,
            ),
            MctsType::Evolved => perform_mcts_evolved(
                state,
                side_one_options.clone(),
                side_two_options.clone(),
                self.search_time,
            ),
        };

        // Get the move with highest visits
        let moves = if is_player_one {
            &result.s1
        } else {
            &result.s2
        };
        let total_visits: i64 = moves.iter().map(|m| m.visits).sum();

        let move_weights = if total_visits > 0 {
            moves
                .iter()
                .map(|m| m.visits as f32 / total_visits as f32)
                .collect()
        } else {
            vec![1.0 / moves.len() as f32; moves.len()]
        };

        let best_idx = moves
            .iter()
            .enumerate()
            .max_by_key(|(_, m)| m.visits)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let options = if is_player_one {
            side_one_options
        } else {
            side_two_options
        };

        PlayerChoice {
            chosen_move: options[best_idx].clone(),
            move_weights,
        }
    }
}

pub struct RandomPlayer;

impl Player for RandomPlayer {
    fn choose_move(&mut self, state: &mut State, is_player_one: bool, _: u32) -> PlayerChoice {
        let (options, _) = if is_player_one {
            state.get_all_options()
        } else {
            let (s2, s1) = state.get_all_options();
            (s2, s1)
        };

        if options.is_empty() {
            return PlayerChoice {
                chosen_move: MoveChoice::None,
                move_weights: vec![1.0],
            };
        }

        let uniform_weight = 1.0 / options.len() as f32;
        let weights = vec![uniform_weight; options.len()];

        let mut rng = rand::thread_rng();
        let chosen_idx = rng.gen_range(0..options.len());

        PlayerChoice {
            chosen_move: options[chosen_idx].clone(),
            move_weights: weights,
        }
    }
}

pub struct MctsPlayerPruned {
    search_time: Duration,
    model_path: String,
    model_cache: Mutex<Option<Arc<JointNetwork>>>,
}

impl MctsPlayerPruned {
    pub fn new(search_time: Duration, model_path: &str) -> Self {
        Self {
            search_time,
            model_path: model_path.to_string(),
            model_cache: Mutex::new(None),
        }
    }

    // Initialize model if not already cached
    fn get_model(&self) -> Option<Arc<JointNetwork>> {
        let mut model_guard = self.model_cache.lock().unwrap();

        if model_guard.is_none() {
            let device = Device::Cpu;
            match JointNetwork::new(&self.model_path, device) {
                Ok(model) => {
                    *model_guard = Some(Arc::new(model));
                }
                Err(e) => {
                    eprintln!("Failed to load joint network model: {}", e);
                    return None;
                }
            }
        }

        model_guard.as_ref().map(Arc::clone)
    }
}

impl Player for MctsPlayerPruned {
    fn choose_move(
        &mut self,
        state: &mut State,
        is_player_one: bool,
        _turn_count: u32,
    ) -> PlayerChoice {
        // Get the model
        let model = match self.get_model() {
            Some(m) => m,
            None => {
                eprintln!("WARNING: Model failed to load - using fallback strategy");
                // Fallback if model loading fails - use a default strategy
                return self.fallback_strategy(state, is_player_one);
            }
        };

        let (side_one_options, side_two_options) = state.get_all_options();

        // Perform MCTS with value network guidance
        let result = perform_mcts_pruned_batched(
            state,
            side_one_options.clone(),
            side_two_options.clone(),
            self.search_time,
            model,
        );

        // Get the moves for the current player
        let moves = if is_player_one {
            &result.s1
        } else {
            &result.s2
        };

        // Calculate move weights based on visit counts
        let total_visits: i64 = moves.iter().map(|m| m.visits).sum();
        let move_weights = if total_visits > 0 {
            moves
                .iter()
                .map(|m| m.visits as f32 / total_visits as f32)
                .collect()
        } else {
            vec![1.0 / moves.len() as f32; moves.len()]
        };

        // Choose the move with highest visit count
        let best_idx = moves
            .iter()
            .enumerate()
            .max_by_key(|(_, m)| m.visits)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let chosen_move = moves[best_idx].move_choice.clone();

        PlayerChoice {
            chosen_move,
            move_weights,
        }
    }
}

impl MctsPlayerPruned {
    fn fallback_strategy(&self, state: &mut State, is_player_one: bool) -> PlayerChoice {
        // Simple fallback - choose first legal move
        let (side_one_options, side_two_options) = state.get_all_options();

        let options = if is_player_one {
            side_one_options
        } else {
            side_two_options
        };

        if options.is_empty() {
            // No legal moves - return None
            PlayerChoice {
                chosen_move: MoveChoice::None,
                move_weights: vec![1.0],
            }
        } else {
            // Choose first available move
            let move_weights = vec![1.0 / options.len() as f32; options.len()];
            PlayerChoice {
                chosen_move: options[0].clone(),
                move_weights,
            }
        }
    }
}

pub struct MctsPlayerPN {
    search_time: Duration,
    model_path: String,
    model_cache: std::sync::Mutex<Option<Arc<PolicyNetwork>>>,
    top_k_moves: usize,
}

impl MctsPlayerPN {
    pub fn new(search_time: Duration, model_path: &str, top_k_moves: Option<&usize>) -> Self {
        Self {
            search_time,
            model_path: model_path.to_string(),
            model_cache: std::sync::Mutex::new(None),
            top_k_moves: *top_k_moves.unwrap_or(&DEFAULT_TOP_K_MOVES),
        }
    }

    // Initialize model if not already cached
    fn get_model(&self) -> Option<Arc<PolicyNetwork>> {
        let mut model_guard = self.model_cache.lock().unwrap();

        if model_guard.is_none() {
            let device = Device::Cpu;
            match PolicyNetwork::new(&self.model_path, device) {
                Ok(model) => {
                    *model_guard = Some(Arc::new(model));
                }
                Err(e) => {
                    eprintln!("Failed to load policy network model: {}", e);
                    return None;
                }
            }
        }

        model_guard.as_ref().map(Arc::clone)
    }

    // Fallback strategy if model loading fails
    fn fallback_strategy(&self, state: &mut State, is_player_one: bool) -> PlayerChoice {
        // Simple fallback - choose first legal move
        let (side_one_options, side_two_options) = state.get_all_options();

        let options = if is_player_one {
            side_one_options
        } else {
            side_two_options
        };

        if options.is_empty() {
            // No legal moves - return None
            PlayerChoice {
                chosen_move: MoveChoice::None,
                move_weights: vec![1.0],
            }
        } else {
            // Choose first available move
            let move_weights = vec![1.0 / options.len() as f32; options.len()];
            PlayerChoice {
                chosen_move: options[0].clone(),
                move_weights,
            }
        }
    }
}

impl Player for MctsPlayerPN {
    fn choose_move(
        &mut self,
        state: &mut State,
        is_player_one: bool,
        _turn_count: u32,
    ) -> PlayerChoice {
        // Get the model
        let model = match self.get_model() {
            Some(m) => m,
            None => {
                // Fallback if model loading fails
                return self.fallback_strategy(state, is_player_one);
            }
        };

        let (side_one_options, side_two_options) = state.get_all_options();

        // Perform MCTS with policy network guidance
        let result = perform_mcts_pn_batched(
            state,
            side_one_options.clone(),
            side_two_options.clone(),
            self.search_time,
            model,
            self.top_k_moves,
        );

        // Get the moves for the current player
        let moves = if is_player_one {
            &result.s1
        } else {
            &result.s2
        };

        // Calculate move weights based on visit counts
        let total_visits: i64 = moves.iter().map(|m| m.visits).sum();
        let move_weights = if total_visits > 0 {
            moves
                .iter()
                .map(|m| m.visits as f32 / total_visits as f32)
                .collect()
        } else {
            vec![1.0 / moves.len() as f32; moves.len()]
        };

        // Choose the move with highest visit count
        let best_idx = moves
            .iter()
            .enumerate()
            .max_by_key(|(_, m)| m.visits)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let chosen_move = moves[best_idx].move_choice.clone();

        PlayerChoice {
            chosen_move,
            move_weights,
        }
    }
}
