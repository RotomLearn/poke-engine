use crate::mcts::perform_mcts;
use crate::mcts_az::{perform_mcts_az_with_forced, AZParams, MctsSideResultAZ, NeuralNet};
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

pub struct MctsPlayer {
    search_time: Duration,
}

impl MctsPlayer {
    pub fn new(search_time: Duration) -> Self {
        Self { search_time }
    }
}

impl Player for MctsPlayer {
    fn choose_move(&mut self, state: &mut State, is_player_one: bool, _: u32) -> PlayerChoice {
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

// Updated AlphaZero MCTS Player
pub struct MctsPlayerAZ {
    search_time: Duration,
    model_path: String,
    // Temperature parameters
    use_temperature: bool,
    temperature_threshold: u32,
    // Dirichlet noise parameters
    dirichlet_alpha: f32,
    dirichlet_weight: f32,
    // Thread-safe model cache with lazy initialization
    model_cache: Mutex<Option<Arc<NeuralNet>>>,
}

impl MctsPlayerAZ {
    pub fn new(search_time: Duration, model_path: &str) -> Self {
        Self {
            search_time,
            model_path: model_path.to_string(),
            use_temperature: true,
            temperature_threshold: 15, // first 15 turns use temperature = 1.0
            dirichlet_alpha: 0.3,      // base alpha value
            dirichlet_weight: 0.25,    // 25% noise, 75% network prior
            model_cache: Mutex::new(None),
        }
    }

    // Creating an evaluation-specific version
    pub fn new_for_eval(search_time: Duration, model_path: &str) -> Self {
        Self {
            search_time,
            model_path: model_path.to_string(),
            use_temperature: false, // always use greedy selection in evaluation
            temperature_threshold: 0,
            dirichlet_alpha: 0.0,  // no noise during evaluation
            dirichlet_weight: 0.0, // no noise during evaluation
            model_cache: Mutex::new(None),
        }
    }

    // Add a method to customize temperature and noise settings
    pub fn with_settings(
        mut self,
        use_temperature: bool,
        temperature_threshold: u32,
        dirichlet_alpha: f32,
        dirichlet_weight: f32,
    ) -> Self {
        self.use_temperature = use_temperature;
        self.temperature_threshold = temperature_threshold;
        self.dirichlet_alpha = dirichlet_alpha;
        self.dirichlet_weight = dirichlet_weight;
        self
    }

    // Initialize model if not already cached
    fn get_model(&self) -> Arc<NeuralNet> {
        let mut model_guard = self.model_cache.lock().unwrap();

        if model_guard.is_none() {
            let device = Device::Cpu;
            let model = NeuralNet::new(&self.model_path, device)
                .expect("Failed to load neural network model");
            *model_guard = Some(Arc::new(model));
        }

        Arc::clone(model_guard.as_ref().unwrap())
    }

    // Helper function to sample move with temperature
    fn sample_move_with_temperature(
        &self,
        moves: &[MctsSideResultAZ],
        turn_count: u32,
        rng: &mut impl Rng,
    ) -> usize {
        // If not using temperature or past threshold, use greedy selection
        if !self.use_temperature || turn_count > self.temperature_threshold {
            return moves
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.visits.cmp(&b.visits))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        // Get visit counts
        let visits: Vec<i64> = moves.iter().map(|m| m.visits).collect();

        // Avoid division by zero by checking if there are any visits
        if visits.iter().sum::<i64>() == 0 {
            return rng.gen_range(0..moves.len());
        }

        // Apply temperature = 1.0 by using raw visit counts as weights
        let weights: Vec<f64> = visits.iter().map(|&v| v as f64).collect();

        // Sample according to the probability distribution
        match WeightedIndex::new(&weights) {
            Ok(dist) => dist.sample(rng),
            Err(_) => rng.gen_range(0..moves.len()), // Fallback if weighted sampling fails
        }
    }
}

impl Player for MctsPlayerAZ {
    fn choose_move(
        &mut self,
        state: &mut State,
        is_player_one: bool,
        turn_count: u32,
    ) -> PlayerChoice {
        // Get the cached model or load it if not already loaded
        let model = self.get_model();

        let mut rng = rand::thread_rng();
        let (side_one_options, side_two_options) = state.get_all_options();

        // Setup AlphaZero parameters with adaptive alpha
        let az_params = if self.dirichlet_alpha > 0.0 && self.dirichlet_weight > 0.0 {
            Some(AZParams {
                dirichlet_alpha_base: self.dirichlet_alpha,
                use_adaptive_alpha: true, // Enable adaptive scaling
                dirichlet_weight: self.dirichlet_weight,
            })
        } else {
            None
        };

        // Use the version with parameters
        let result = perform_mcts_az_with_forced(
            state,
            side_one_options.clone(),
            side_two_options.clone(),
            self.search_time,
            model.clone(),
            az_params,
        );

        // Get the move options
        let moves = if is_player_one {
            &result.s1
        } else {
            &result.s2
        };

        // Calculate normalized visit counts (for the policy)
        let total_visits: i64 = moves.iter().map(|m| m.visits).sum();
        let move_weights = if total_visits > 0 {
            moves
                .iter()
                .map(|m| m.visits as f32 / total_visits as f32)
                .collect()
        } else {
            vec![1.0 / moves.len() as f32; moves.len()]
        };

        // Choose move using temperature sampling
        let best_idx = self.sample_move_with_temperature(moves, turn_count, &mut rng);

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
