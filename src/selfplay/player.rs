use crate::mcts::perform_mcts;
use crate::mcts_az::{perform_mcts_az, NeuralNet};
use crate::state::{MoveChoice, State};
use rand::Rng;
use std::sync::Arc;
use std::time::Duration;
use tch::Device;

pub struct PlayerChoice {
    pub chosen_move: MoveChoice,
    pub move_weights: Vec<f32>, // Will be normalized visit counts for MCTS, uniform for random
}

pub trait Player {
    fn choose_move(&mut self, state: &mut State, is_player_one: bool) -> PlayerChoice;
}

pub struct MctsPlayer {
    search_time: Duration,
}

impl MctsPlayer {
    pub fn new(search_time: Duration) -> Self {
        Self { search_time }
    }
}

impl Player for MctsPlayer {
    fn choose_move(&mut self, state: &mut State, is_player_one: bool) -> PlayerChoice {
        let (side_one_options, side_two_options) = state.get_all_options();

        let result = perform_mcts(
            state,
            side_one_options.clone(),
            side_two_options.clone(),
            self.search_time,
        );

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
    fn choose_move(&mut self, state: &mut State, is_player_one: bool) -> PlayerChoice {
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

pub struct MctsPlayerAZ {
    search_time: Duration,
    model_path: String,
}

impl MctsPlayerAZ {
    pub fn new(search_time: Duration, model_path: &str) -> Self {
        Self {
            search_time,
            model_path: model_path.to_string(),
        }
    }
}

impl Player for MctsPlayerAZ {
    fn choose_move(&mut self, state: &mut State, is_player_one: bool) -> PlayerChoice {
        let device = Device::Cpu; // or Device::Cuda(0) for GPU
        let model = match NeuralNet::new(&self.model_path, device) {
            Ok(model) => Arc::new(model),
            Err(e) => panic!("Failed to load model: {}", e),
        };
        let (side_one_options, side_two_options) = state.get_all_options();

        let result = perform_mcts_az(
            state,
            side_one_options.clone(),
            side_two_options.clone(),
            self.search_time,
            model.clone(),
        );

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
