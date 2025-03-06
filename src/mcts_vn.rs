#![allow(non_snake_case)]
use crate::evaluate::evaluate as calc_heuristic_evaluation;
use crate::generate_instructions::generate_instructions_from_move_pair;
use crate::instruction::StateInstructions;
use crate::observation::generate_observation;
use crate::state::{MoveChoice, SideReference, State};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::thread_rng;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::error::Error; // Add this import
use std::sync::Arc;
use std::time::Duration;
use tch::{CModule, Device, Tensor};

// Constants for UCB formula
const C_UCB: f32 = 2.0;

// Constants for hybrid evaluation
const NEURAL_NET_DEPTH_THRESHOLD: i32 = 6; // Use neural net only after this search depth
const MAX_NEURAL_NET_CALLS_PER_ITERATION: usize = 100000; // Limit neural net calls per iteration
const MODEL_IMPORTANCE: f32 = 200.0; // Importance of neural network evaluation

pub struct ValueNetwork {
    model: CModule,
    device: Device,
}

impl ValueNetwork {
    pub fn new(model_path: &str, device: Device) -> Result<Self, Box<dyn Error>> {
        // Set threading configuration
        tch::set_num_threads(1);
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("MKL_NUM_THREADS", "1");
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");

        // Load the model
        let model = CModule::load(model_path)?;
        Ok(ValueNetwork { model, device })
    }

    pub fn evaluate_batch(&self, states: &[State]) -> Vec<f32> {
        let batch_size = states.len();

        // Pre-allocate all buffers at full size to avoid reallocations
        let mut weather = Vec::with_capacity(batch_size * 14);
        let mut trick_room = Vec::with_capacity(batch_size * 7);
        let mut our_side_conditions = Vec::with_capacity(batch_size * 76);
        let mut opp_side_conditions = Vec::with_capacity(batch_size * 76);
        let mut our_volatile_status = Vec::with_capacity(batch_size * 104);
        let mut opp_volatile_status = Vec::with_capacity(batch_size * 104);
        let mut our_boosts = Vec::with_capacity(batch_size * 78);
        let mut opp_boosts = Vec::with_capacity(batch_size * 78);
        let mut evaluate = Vec::with_capacity(batch_size);

        // Create a tensor for all Pokémon (batch_size, 12, 104)
        let mut all_pokemon = vec![0.0; batch_size * 12 * 104];

        // Process each state
        for (i, state) in states.iter().enumerate() {
            // Get weather (first 14 elements of observation)
            let obs = generate_observation(state, SideReference::SideOne);
            weather.extend_from_slice(&obs[0..14]);

            // Get trick room (next 7 elements)
            trick_room.extend_from_slice(&obs[14..21]);

            // Get side conditions (next 76 elements for each side)
            our_side_conditions.extend_from_slice(&obs[21..97]);
            opp_side_conditions.extend_from_slice(&obs[97..173]);

            // Get volatile status (next 104 elements for each side)
            our_volatile_status.extend_from_slice(&obs[173..277]);
            opp_volatile_status.extend_from_slice(&obs[277..381]);

            // Get boosts (next 78 elements for each side)
            our_boosts.extend_from_slice(&obs[381..459]);
            opp_boosts.extend_from_slice(&obs[459..537]);

            // Process Pokémon (this assumes a specific format in your observation)
            let pokemon_start = 537; // Adjust if needed

            // Process our 6 Pokémon
            for p in 0..6 {
                let p_start = pokemon_start + p * 104;
                let p_end = p_start + 104;
                let pokemon_slice = if p_end <= obs.len() {
                    &obs[p_start..p_end]
                } else {
                    // Handle case where observation doesn't have all expected fields
                    &[0.0; 104][..]
                };

                // Calculate the position in the flattened tensor
                let tensor_idx = (i * 12 + p) * 104;
                all_pokemon[tensor_idx..tensor_idx + 104].copy_from_slice(pokemon_slice);
            }

            // Process opponent's 6 Pokémon
            for p in 0..6 {
                let p_start = pokemon_start + (p + 6) * 104;
                let p_end = p_start + 104;
                let pokemon_slice = if p_end <= obs.len() {
                    &obs[p_start..p_end]
                } else {
                    // Handle case where observation doesn't have all expected fields
                    &[0.0; 104][..]
                };

                // Calculate the position in the flattened tensor
                let tensor_idx = (i * 12 + p + 6) * 104;
                all_pokemon[tensor_idx..tensor_idx + 104].copy_from_slice(pokemon_slice);
            }

            // Get evaluation value (last element or calculate from state)
            let eval_value = if obs.len() > pokemon_start + 12 * 104 {
                obs[pokemon_start + 12 * 104]
            } else {
                // If evaluation isn't in observation, use a default or calculate
                calc_heuristic_evaluation(state) / 500.0
            };

            evaluate.push(eval_value);
        }

        // Convert to tensors
        let weather_tensor = Tensor::from_slice(&weather)
            .to_device(self.device)
            .view([batch_size as i64, 14]);

        let trick_room_tensor = Tensor::from_slice(&trick_room)
            .to_device(self.device)
            .view([batch_size as i64, 7]);

        let our_side_conditions_tensor = Tensor::from_slice(&our_side_conditions)
            .to_device(self.device)
            .view([batch_size as i64, 76]);

        let opp_side_conditions_tensor = Tensor::from_slice(&opp_side_conditions)
            .to_device(self.device)
            .view([batch_size as i64, 76]);

        let our_volatile_status_tensor = Tensor::from_slice(&our_volatile_status)
            .to_device(self.device)
            .view([batch_size as i64, 104]);

        let opp_volatile_status_tensor = Tensor::from_slice(&opp_volatile_status)
            .to_device(self.device)
            .view([batch_size as i64, 104]);

        let our_boosts_tensor = Tensor::from_slice(&our_boosts)
            .to_device(self.device)
            .view([batch_size as i64, 78]);

        let opp_boosts_tensor = Tensor::from_slice(&opp_boosts)
            .to_device(self.device)
            .view([batch_size as i64, 78]);

        let pokemon_tensor = Tensor::from_slice(&all_pokemon)
            .to_device(self.device)
            .view([batch_size as i64, 12, 104]);

        let evaluate_tensor = Tensor::from_slice(&evaluate)
            .to_device(self.device)
            .view([batch_size as i64, 1]);

        // Forward pass
        let output = self
            .model
            .forward_ts(&[
                weather_tensor,
                trick_room_tensor,
                our_side_conditions_tensor,
                opp_side_conditions_tensor,
                our_volatile_status_tensor,
                opp_volatile_status_tensor,
                our_boosts_tensor,
                opp_boosts_tensor,
                pokemon_tensor,
                evaluate_tensor,
            ])
            .unwrap();

        // Extract values
        let values = match Vec::<f32>::try_from(&output) {
            Ok(v) => v,
            Err(_) => {
                // If conversion fails, extract using a safer but slower method
                let mut result = Vec::with_capacity(batch_size);
                for i in 0..batch_size {
                    result.push(output.get(i as i64).double_value(&[0]) as f32);
                }
                result
            }
        };

        values
    }

    pub fn evaluate(&self, state: &State) -> f32 {
        // Reuse the batch method for simplicity
        let values = self.evaluate_batch(&[state.clone()]);
        values[0]
    }
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub children: HashMap<(usize, usize), Vec<Node>>,
    pub times_visited: i64,
    pub depth: i32, // Track depth in the tree

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: usize,
    pub s2_choice: usize,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Vec<MoveNode>,
    pub s2_options: Vec<MoveNode>,

    // Neural network evaluation data
    pub value_sum: f32,
    pub neural_value: Option<f32>, // Store neural evaluation if performed
    pub heuristic_value: f32,      // Store heuristic evaluation
    pub needs_neural_eval: bool,   // Flag for nodes that need neural evaluation
}

impl Node {
    fn new(s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>, depth: i32) -> Node {
        let s1_options_vec = s1_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();
        let s2_options_vec = s2_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();

        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            depth,
            children: HashMap::new(),
            s1_choice: 0,
            s2_choice: 0,
            s1_options: s1_options_vec,
            s2_options: s2_options_vec,
            value_sum: 0.0,
            neural_value: None,
            heuristic_value: 0.0,
            needs_neural_eval: false,
        }
    }

    fn get_max_depth(&self) -> usize {
        if self.children.is_empty() {
            return 0;
        }

        let mut max_child_depth = 0;
        for (_, children) in &self.children {
            for child in children {
                max_child_depth = max_child_depth.max(child.get_max_depth());
            }
        }
        max_child_depth + 1
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode]) -> usize {
        let mut choice = 0;
        let mut best_ucb = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb = node.ucb(self.times_visited);
            if this_ucb > best_ucb {
                best_ucb = this_ucb;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(&mut self, state: &mut State) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;

        // Print current depth for debugging
        // println!("In selection at depth: {}", self.depth);

        let s1_mc_index = self.maximize_ucb_for_side(&self.s1_options);
        let s2_mc_index = self.maximize_ucb_for_side(&self.s2_options);

        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);

                // Verify child depth
                // println!("Selected child at depth: {}", (*chosen_child).depth);

                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state)
            }
            None => (return_node, s1_mc_index, s2_mc_index),
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Vec<Node>) -> *mut Node {
        let mut rng = thread_rng();
        let weights: Vec<f64> = (*move_vector)
            .iter()
            .map(|x| x.instructions.percentage as f64)
            .collect();
        let dist = WeightedIndex::new(weights).unwrap();
        let chosen_node = &mut (*move_vector)[dist.sample(&mut rng)];
        let chosen_node_ptr = chosen_node as *mut Node;
        chosen_node_ptr
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
    ) -> *mut Node {
        let s1_move = &self.s1_options[s1_move_index].move_choice;
        let s2_move = &self.s2_options[s2_move_index].move_choice;

        // If the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return self as *mut Node;
        }

        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);

        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());

        // Create new child depth
        let child_depth = self.depth + 1;

        for state_instructions in new_instructions.drain(..) {
            state.apply_instructions(&state_instructions.instruction_list);

            // Get options and evaluate with heuristic
            let (s1_options, s2_options) = state.get_all_options();
            let heuristic_value = calc_heuristic_evaluation(state);

            // Determine if this node needs neural evaluation
            let needs_neural_eval = child_depth >= NEURAL_NET_DEPTH_THRESHOLD;

            // Create node
            let mut new_node = Node::new(s1_options, s2_options, child_depth);
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index;
            new_node.s2_choice = s2_move_index;
            new_node.heuristic_value = heuristic_value;
            new_node.needs_neural_eval = needs_neural_eval;

            // Revert state for next iteration
            state.reverse_instructions(&new_node.instructions.instruction_list);

            this_pair_vec.push(new_node);
        }

        // Sample a node from the new instruction list
        let new_node_ptr = self.sample_node(&mut this_pair_vec);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);
        self.children
            .insert((s1_move_index, s2_move_index), this_pair_vec);

        new_node_ptr
    }

    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State, neural: bool) {
        if neural {
            self.times_visited += MODEL_IMPORTANCE as i64;
        } else {
            self.times_visited += 1;
        }
        self.value_sum += score;

        if self.root {
            return;
        }

        let parent_s1_movenode = &mut (*self.parent).s1_options[self.s1_choice];
        if neural {
            parent_s1_movenode.total_score += MODEL_IMPORTANCE * score;
            parent_s1_movenode.visits += MODEL_IMPORTANCE as i64;
        } else {
            parent_s1_movenode.total_score += score;
            parent_s1_movenode.visits += 1;
        }

        let parent_s2_movenode = &mut (*self.parent).s2_options[self.s2_choice];
        if neural {
            parent_s2_movenode.total_score += MODEL_IMPORTANCE * (1.0 - score);
            parent_s2_movenode.visits += MODEL_IMPORTANCE as i64;
        } else {
            parent_s2_movenode.total_score += 1.0 - score;
            parent_s2_movenode.visits += 1;
        }

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state, neural);
    }

    pub fn evaluate(
        &mut self,
        state: &State,
        network: Option<&ValueNetwork>,
        neural_eval_counter: &mut usize,
    ) -> (f32, bool) {
        // Print debug info
        // println!(
        //     "Evaluating at depth {}, needs_neural: {}, times_visited: {}, network available: {}",
        //     self.depth,
        //     self.needs_neural_eval,
        //     self.times_visited,
        //     network.is_some()
        // );

        let battle_is_over = state.battle_is_over();
        if battle_is_over != 0.0 {
            return if battle_is_over == -1.0 {
                (0.0, false)
            } else {
                (battle_is_over, false)
            };
        }

        // Check if we should use neural network evaluation
        let should_use_neural = self.needs_neural_eval
            && self.needs_neural_eval
            && network.is_some()
            && *neural_eval_counter < MAX_NEURAL_NET_CALLS_PER_ITERATION
            && (self.depth >= NEURAL_NET_DEPTH_THRESHOLD || self.root);

        if should_use_neural {
            // Debug print when using neural evaluation
            // println!("Using neural eval at depth {}", self.depth);

            // If we already have a neural evaluation, use it
            if let Some(value) = self.neural_value {
                return (value, true);
            }

            // Otherwise, evaluate with the neural network
            if let Some(net) = network {
                *neural_eval_counter += 1;
                let mapped_value = net.evaluate(state);
                self.neural_value = Some(mapped_value);
                return (mapped_value, true);
            }
        }

        // If we reach here, use the heuristic value
        // Convert heuristic to a value between 0 and 1
        (sigmoid(self.heuristic_value), false)
    }
}

// Sigmoid function to normalize heuristic values
fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: i64,
}

impl MoveNode {
    pub fn ucb(&self, parent_visits: i64) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let score = (self.total_score / self.visits as f32)
            + (C_UCB * (parent_visits as f32).ln() / self.visits as f32).sqrt();
        score
    }

    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        self.total_score / self.visits as f32
    }
}

fn do_mcts(
    root_node: &mut Node,
    state: &mut State,
    network: Option<&ValueNetwork>,
    neural_eval_counter: &mut usize,
) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move) };
    let (evaluation, neural) = unsafe { (*new_node).evaluate(state, network, neural_eval_counter) };
    unsafe { (*new_node).backpropagate(evaluation, state, neural) }
}

#[derive(Clone)]
pub struct MctsSideResultVN {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: i64,
}

impl MctsSideResultVN {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        self.total_score / self.visits as f32
    }
}

pub struct MctsResultVN {
    pub s1: Vec<MctsSideResultVN>,
    pub s2: Vec<MctsSideResultVN>,
    pub iteration_count: i64,
    pub max_depth: usize,
    pub neural_evals_count: usize,
}

pub fn perform_mcts_vn(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    network: Option<Arc<ValueNetwork>>,
) -> MctsResultVN {
    // Create root node at depth 0
    let mut root_node = Node::new(side_one_options, side_two_options, 0);
    root_node.root = true;

    // Evaluate root with heuristic
    root_node.heuristic_value = calc_heuristic_evaluation(state);

    // Count total neural evaluations
    let mut total_neural_evals = 0;

    let start_time = std::time::Instant::now();
    while start_time.elapsed() < max_time {
        // Reset counter for each batch of iterations
        let mut neural_eval_counter = 0;

        for _ in 0..100 {
            do_mcts(
                &mut root_node,
                state,
                network.as_ref().map(|n| n.as_ref()),
                &mut neural_eval_counter,
            );
        }

        total_neural_evals += neural_eval_counter;

        if root_node.times_visited == 10_000_000 {
            break;
        }
    }

    let max_depth = root_node.get_max_depth();

    MctsResultVN {
        s1: root_node
            .s1_options
            .iter()
            .map(|v| MctsSideResultVN {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .iter()
            .map(|v| MctsSideResultVN {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
        max_depth,
        neural_evals_count: total_neural_evals,
    }
}
